using System;
using System.Text.Json;
using System.Threading.Tasks;
using Polytopia.Data;
using UnityEngine;

namespace PolyfishAI.src
{
    public static class PolyfishBot
    {
        public static bool IsBotEnabled = true;
        private static bool _isFetchingMove = false;
        private static float _pollTimer = 0f;
        private const float POLL_INTERVAL = 1.0f; // Check every 1 second

        public static void Update()
        {
            if (Input.GetKeyDown(KeyCode.B))
            {
                IsBotEnabled = !IsBotEnabled;
                PolyfishPlugin.Logger.LogInfo($"[Bot] Bot mode toggled: {IsBotEnabled}");
            }

            if (!IsBotEnabled) return;
            if (_isFetchingMove) return;

            _pollTimer += Time.unscaledDeltaTime;
            if (_pollTimer < POLL_INTERVAL) return;
            _pollTimer = 0f;

            var gameState = GameManager.GameState;
            if (gameState == null || gameState.Map == null) return;

            // Optional: Check if it's our turn
            // int currentPlayerId = gameState.CurrentPlayerIndex;
            
            _isFetchingMove = true;
            PolyfishPlugin.Logger.LogInfo("[Bot] Requesting next move from backend...");
            
            Task.Run(async () =>
            {
                try
                {
                    await FetchAndExecuteMove(gameState);
                }
                catch (Exception ex)
                {
                    PolyfishPlugin.Logger.LogError($"[Bot] Error fetching/executing move: {ex}");
                }
                finally
                {
                    _isFetchingMove = false;
                }
            });
        }

        private static async Task FetchAndExecuteMove(GameState gameState)
        {
            // First, save the state to ensure backend is in sync
            await PolyfishPlugin.API.SaveStateAsync(gameState);

            // Request the next move
            // We use standard HttpClient here since PolyfishAPI doesn't expose a direct GET/POST with return yet,
            // or we can add an endpoint to PolyfishAPI.
            string responseJson = await PolyfishAPI_GetMoveAsync();
            if (string.IsNullOrEmpty(responseJson)) return;

            PolyfishPlugin.RunOnMainThread(() =>
            {
                try
                {
                    Dumper.Dump(); ExecuteMoveFromJson(responseJson, gameState);
                }
                catch (Exception ex)
                {
                    PolyfishPlugin.Logger.LogError($"[Bot] Failed to execute move: {ex}");
                }
            });
        }

        private static async Task<string> PolyfishAPI_GetMoveAsync()
        {
            using var client = new System.Net.Http.HttpClient();
            var payload = new { iterations = 400, dry_run = true };
            var content = new System.Net.Http.StringContent(JsonSerializer.Serialize(payload), System.Text.Encoding.UTF8, "application/json");
            
            var response = await client.PostAsync("http://localhost:3000/autostep", content);
            if (!response.IsSuccessStatusCode) return null;
            return await response.Content.ReadAsStringAsync();
        }

        private static void ExecuteMoveFromJson(string json, GameState gameState)
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            
            if (!root.TryGetProperty("bestMove", out var bestMove))
            {
                PolyfishPlugin.Logger.LogWarning("[Bot] No bestMove found in response.");
                return;
            }

            int moveType = bestMove.GetProperty("moveType").GetInt32();
            int size = gameState.Settings.MapSize;

            int srcIdx = bestMove.TryGetProperty("src", out var srcProp) ? srcProp.GetInt32() : -1;
            int targetIdx = bestMove.TryGetProperty("target", out var tgtProp) ? tgtProp.GetInt32() : -1;
            int type = bestMove.TryGetProperty("type", out var typeProp) ? typeProp.GetInt32() : -1;
            int reward = bestMove.TryGetProperty("_reward", out var rewardProp) ? rewardProp.GetInt32() : -1;
            int playerId = gameState.CurrentPlayerIndex; // Default to current player

            CommandBase cmd = null;

            // Map Index to WorldCoordinates
            WorldCoordinates GetCoords(int idx)
            {
                int x = idx % size;
                int y = idx / size;
                return new WorldCoordinates(x, y);
            }

            PolyfishPlugin.Logger.LogInfo($"[Bot] Creating command: moveType={moveType}, src={srcIdx}, target={targetIdx}");

            // Instantiate IL2CPP Commands
            // We use default constructor and set properties
            switch (moveType)
            {
                case 1: // Step
                    var move = new MoveCommand();
                    move.From = GetCoords(srcIdx);
                    move.To = GetCoords(targetIdx);
                    move.PlayerId = (byte)playerId;
                    cmd = move;
                    break;
                case 2: // Attack
                    var attack = new AttackCommand();
                    attack.Origin = GetCoords(srcIdx);
                    attack.Target = GetCoords(targetIdx);
                    attack.PlayerId = (byte)playerId;
                    cmd = attack;
                    break;
                case 3: // Ability
                    // Needs specific ability command
                    PolyfishPlugin.Logger.LogWarning("[Bot] Ability commands not fully mapped yet.");
                    break;
                case 4: // Summon/Train
                    var train = new TrainCommand();
                    train.Coordinates = GetCoords(srcIdx);
                    train.Type = (UnitData.Type)type;
                    train.PlayerId = (byte)playerId;
                    cmd = train;
                    break;
                case 5: // Harvest
                case 6: // Build
                    var build = new BuildCommand();
                    build.Coordinates = GetCoords(targetIdx != -1 ? targetIdx : srcIdx);
                    build.Type = (ImprovementData.Type)type;
                    build.PlayerId = (byte)playerId;
                    cmd = build;
                    break;
                case 7: // Research
                    var research = new ResearchCommand();
                    research.Type = (TechData.Type)type;
                    research.PlayerId = (byte)playerId;
                    cmd = research;
                    break;
                case 8: // Capture / ExamineRuins
                    var capture = new CaptureCommand();
                    capture.Coordinates = GetCoords(srcIdx);
                    capture.PlayerId = (byte)playerId;
                    cmd = capture;
                    break;
                case 9: // Reward
                    var cityReward = new CityRewardCommand();
                    cityReward.Coordinates = GetCoords(targetIdx != -1 ? targetIdx : srcIdx);
                    cityReward.Reward = (CityReward)type;
                    cityReward.PlayerId = (byte)playerId;
                    cmd = cityReward;
                    break;
                case 10: // EndTurn
                    cmd = new EndTurnCommand();
                    cmd.PlayerId = (byte)playerId;
                    break;
                default:
                    PolyfishPlugin.Logger.LogWarning($"[Bot] Unknown moveType: {moveType}");
                    break;
            }

            if (cmd != null)
            {
                PolyfishPlugin.Logger.LogInfo($"[Bot] Injecting command: {cmd.GetType().Name}");
                
                // Try executing it via InteractionBar's Method_Internal_Void_CommandBase_0 or similar
                var interactionBar = UnityEngine.Object.FindObjectOfType<InteractionBar>();
                if (interactionBar != null)
                {
                    try 
                    {
                        // Some commands are handled by OnPopupAccepted
                        interactionBar.OnPopupAccepted(cmd);
                    }
                    catch (Exception)
                    {
                        PolyfishPlugin.Logger.LogWarning("[Bot] Failed to use OnPopupAccepted, attempting alternative...");
                    }
                }
                else
                {
                    PolyfishPlugin.Logger.LogWarning("[Bot] InteractionBar not found!");
                }
            }
        }
    }
}
