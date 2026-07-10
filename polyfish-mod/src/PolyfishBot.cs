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
        public static bool UseRandomMoves = true; // Set to true to bypass slow MCTS for testing
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

            // Automatically dismiss any popups that block the bot!
            if (PopupManager.PopupShowing)
            {
                try
                {
                    PopupManager.SkipCurrentPopup();
                }
                catch (Exception) { }
                return; // DO NOT fetch or play moves while a popup is showing!
            }

            if (_isFetchingMove) return;

            _pollTimer += Time.unscaledDeltaTime;
            if (_pollTimer < POLL_INTERVAL) return;
            _pollTimer = 0f;

            var gm = UnityEngine.Object.FindObjectOfType<GameManager>();
            if (gm == null || gm.client == null) return;
            // PolyfishPlugin.Logger.LogInfo($"[Bot] IsWaitingForCommand = {gm.client.IsWaitingForCommand}");
            // We ignore IsWaitingForCommand because it's false even when it's our turn.

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
                    ExecuteMoveFromJson(responseJson, gameState);
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
            var payload = new { iterations = 400, dry_run = true, random = UseRandomMoves };
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
            int playerId = gameState.PlayerStates[gameState.CurrentPlayerIndex].Id; // Use actual Player ID, not list index

            CommandBase cmd = null;

            // Map Index to WorldCoordinates
            WorldCoordinates GetCoords(int idx)
            {
                int x = idx % size;
                int y = idx / size;
                return new WorldCoordinates(x, y);
            }

            uint unitId = 0;
            if (srcIdx >= 0 && srcIdx < gameState.Map.Tiles.Length)
            {
                var unit = gameState.Map.Tiles[srcIdx].unit;
                if (unit != null) unitId = unit.id;
            }

            PolyfishPlugin.Logger.LogInfo($"[Bot] Creating command: moveType={moveType}, src={srcIdx}, target={targetIdx}, unitId={unitId}");

            // Instantiate IL2CPP Commands
            // We use default constructor and set properties
            switch (moveType)
            {
                case 1: // Step
                    var move = new MoveCommand();
                    move.From = GetCoords(srcIdx);
                    move.To = GetCoords(targetIdx);
                    move.PlayerId = (byte)playerId;
                    move.UnitId = unitId;
                    cmd = move;
                    break;
                case 2: // Attack
                    var attack = new AttackCommand();
                    attack.Origin = GetCoords(srcIdx);
                    attack.Target = GetCoords(targetIdx);
                    attack.PlayerId = (byte)playerId;
                    attack.UnitId = unitId;
                    cmd = attack;
                    break;
                case 3: // Ability
                    int abilityType = bestMove.TryGetProperty("type", out var typeProp2) ? typeProp2.GetInt32() : 0;
                    switch(abilityType)
                    {
                        case 1: // BurnForest
                            var burn = new BuildCommand();
                            burn.Coordinates = GetCoords(srcIdx);
                            burn.Type = ImprovementData.Type.BurnForest;
                            burn.PlayerId = (byte)playerId;
                            cmd = burn;
                            break;
                        case 2: // ClearForest
                            var clear = new BuildCommand();
                            clear.Coordinates = GetCoords(srcIdx);
                            clear.Type = ImprovementData.Type.ClearForest;
                            clear.PlayerId = (byte)playerId;
                            cmd = clear;
                            break;
                        case 3: // GrowForest
                            var grow = new BuildCommand();
                            grow.Coordinates = GetCoords(srcIdx);
                            grow.Type = ImprovementData.Type.GrowForest;
                            grow.PlayerId = (byte)playerId;
                            cmd = grow;
                            break;
                        case 7: // Recover
                            var recover = new RecoverCommand((byte)playerId, GetCoords(srcIdx));
                            cmd = recover;
                            break;
                        case 8: // Disband
                            var disband = new DisbandCommand((byte)playerId, GetCoords(srcIdx));
                            cmd = disband;
                            break;
                        case 9: // HealOthers
                            var heal = new HealOthersCommand((byte)playerId, GetCoords(srcIdx));
                            cmd = heal;
                            break;
                        case 16: // Promote
                            var promote = new PromoteCommand((byte)playerId, GetCoords(srcIdx));
                            cmd = promote;
                            break;
                        case 4: // Destroy
                            var destroy = new DestroyCommand((byte)playerId, GetCoords(srcIdx));
                            cmd = destroy;
                            break;
                        case 6: // Convert
                            var convert = new AttackCommand();
                            convert.Origin = GetCoords(srcIdx);
                            convert.Target = GetCoords(targetIdx);
                            convert.PlayerId = (byte)playerId;
                            convert.UnitId = unitId;
                            cmd = convert;
                            break;
                        case 13: // FreezeArea
                            var freeze = new FreezeAreaCommand((byte)playerId, GetCoords(srcIdx));
                            cmd = freeze;
                            break;
                        case 17: // BreakIce
                            var breakIce = new BreakIceCommand((byte)playerId, GetCoords(srcIdx));
                            cmd = breakIce;
                            break;
                        default:
                            PolyfishPlugin.Logger.LogWarning($"[Bot] Ability type {abilityType} not mapped yet!");
                            break;
                    }
                    break;
                case 4: // Summon/Train/Upgrade
                    if (gameState.Map.Tiles[srcIdx].unit != null)
                    {
                        var upgrade = new UpgradeCommand((byte)playerId, (UnitData.Type)type, GetCoords(srcIdx));
                        cmd = upgrade;
                    }
                    else
                    {
                        var train = new TrainCommand();
                        train.Coordinates = GetCoords(srcIdx);
                        train.Type = (UnitData.Type)type;
                        train.PlayerId = (byte)playerId;
                        cmd = train;
                    }
                    break;
                case 5: // Harvest
                    var harvestCmd = new BuildCommand();
                    int hTargetIdx = targetIdx != -1 ? targetIdx : srcIdx;
                    harvestCmd.Coordinates = GetCoords(hTargetIdx);
                    var hResType = gameState.Map.Tiles[hTargetIdx].resource?.type ?? ResourceData.Type.None;
                    ImprovementData.Type hImpType = ImprovementData.Type.None;
                    if (hResType == ResourceData.Type.Fruit) hImpType = ImprovementData.Type.HarvestFruit;
                    else if (hResType == ResourceData.Type.Game) hImpType = ImprovementData.Type.Hunting;
                    else if (hResType == ResourceData.Type.Fish) hImpType = ImprovementData.Type.Fishing;
                    else if (hResType == ResourceData.Type.Whale) hImpType = ImprovementData.Type.WhaleHunting;
                    else if (hResType == ResourceData.Type.Spores) hImpType = ImprovementData.Type.HarvestSpores;
                    else if (hResType == ResourceData.Type.Starfish) hImpType = ImprovementData.Type.StarFishing;
                    harvestCmd.Type = hImpType;
                    harvestCmd.PlayerId = (byte)playerId;
                    cmd = harvestCmd;
                    break;
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
                    capture.UnitId = unitId;
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
                var gm = UnityEngine.Object.FindObjectOfType<GameManager>();
                if (gm != null && gm.client != null)
                {
                    PolyfishPlugin.Logger.LogInfo($"[Bot] Injecting command via Client: {cmd.GetType().Name}");
                    gm.client.SendCommand(cmd);
                    PolyfishPlugin.Logger.LogInfo($"[Bot] Sent command successfully.");
                }
                else
                {
                    PolyfishPlugin.Logger.LogWarning("[Bot] GameManager or client not found! Cannot inject command.");
                }
            }
        }
    }
}
