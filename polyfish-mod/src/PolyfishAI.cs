using PolyMod.Api;
using BepInEx.Logging;
using HarmonyLib;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.EventSystems;
using TMPro;
using Polytopia.Data;

namespace PolyfishAI.src
{
    public class PolyfishPlugin : PolyScript
    {
        public static PolyfishAPI API { get; private set; } = null!;
        public static new ManualLogSource Logger { get; private set; } = null!;
        public static string LastExitButtonPath { get; set; } = "Canvas/HUDScreen/ButtonBar/NextTurnButton";

        public static int TargetExplorerCount = 0;
        public static int DiffedExplorerCount = 0;
        public static List<int> ExplorerIndices = new();
        public static Dictionary<int, CommandBase> CommandByIndex = new();
        public static CommandBase? PendingExplorer = null;
        public static HashSet<int> PendingExplorerSnapshot = new();
        public static Dictionary<CommandBase, List<int>> ExplorerRevealedTiles = new();
        public static Dictionary<CommandBase, int> RuinsRewardCache = new();
        public static Dictionary<CommandBase, int> RuinsTechCache = new();
        public static string? InitialGameStateJson = null;
        public static string? CurrentReplayUUID = null;

        public static ReplaySequencer sequencer = null!;

        private static readonly System.Collections.Concurrent.ConcurrentQueue<Action> _mainThreadQueue = new();

        public override void Load()
        {
            Logger = base.Logger;
            base.Logger.LogInfo("PolyfishAI initialized!");
            
            // Overclock the engine to speed up replay extraction
            Application.targetFrameRate = -1; // Uncap framerate
            QualitySettings.vSyncCount = 0;   // Disable VSync
            // Time.timeScale = 40f;             // Run logic 40x faster

            sequencer = new ReplaySequencer();

            // Initialize the bridge to talk to the Rust server (PolyAI)
            API = new PolyfishAPI("http://localhost:3000", base.Logger);

            // Create and apply our Harmony patches
            Harmony.CreateAndPatchAll(typeof(GameHooks));
            Harmony.CreateAndPatchAll(typeof(UILogger));
            Harmony.CreateAndPatchAll(typeof(PolyfishPlugin));
            Harmony.CreateAndPatchAll(typeof(TimelineHotkeys));

            base.Logger.LogInfo("Applied Harmony hooks for PolyfishAI.");
        }

        public static void RunOnMainThread(Action action)
        {
            _mainThreadQueue.Enqueue(action);
        }

        public static string GetGameObjectPath(GameObject obj)
        {
            string path = obj.name;
            Transform parent = obj.transform.parent;
            while (parent != null)
            {
                path = parent.name + "/" + path;
                parent = parent.parent;
            }
            return path;
        }

        public static UIButtonBase? FindButtonByPath(string path)
        {
            // First try active objects
            foreach (var btn in UnityEngine.Object.FindObjectsOfType<UIButtonBase>())
            {
                if (GetGameObjectPath(btn.gameObject) == path)
                    return btn;
            }

            // Then try all objects (including inactive ones)
            foreach (var btn in Resources.FindObjectsOfTypeAll<UIButtonBase>())
            {
                if (GetGameObjectPath(btn.gameObject) == path)
                    return btn;
            }
            return null;
        }

        [HarmonyPatch(typeof(GameManager), "Update")]
        [HarmonyPostfix]
        public static void OnGameManagerUpdate()
        {
            sequencer?.Update();

            if (Input.GetKeyDown(KeyCode.E))
            {
                // TimelineHotkeys.SkipToNextExplorer();
            }

            while (_mainThreadQueue.TryDequeue(out var action))
            {
                try
                {
                    action.Invoke();
                }
                catch (Exception ex)
                {
                    Logger.LogError("Error in MainThreadAction: " + ex);
                }
            }
        }

        public override void Unload()
        {
            base.Logger.LogInfo("PolyfishAI shutting down...");
        }
    }

    /// <summary>
    /// Helper to identify buttons being clicked on screen since UnityExplorer is currently broken.
    /// </summary>
    [HarmonyPatch]
    public class UILogger
    {
        [HarmonyPostfix]
        [HarmonyPatch(typeof(UIButtonBase), "OnPointerClick")]
        public static void OnButtonClicked(UIButtonBase __instance)
        {
            try
            {
                string path = PolyfishPlugin.GetGameObjectPath(__instance.gameObject);
                string text = __instance.GetComponentInChildren<TextMeshProUGUI>()?.text ?? "N/A";
                // PolyfishPlugin.Logger.LogInfo($"[UI] Path: {path} | Logic: {__instance.GetType().Name} | Text: {text}");

                if (text.Equals("Exit", StringComparison.OrdinalIgnoreCase) || text.Equals("QUIT", StringComparison.OrdinalIgnoreCase))
                {
                    PolyfishPlugin.LastExitButtonPath = path;
                    // PolyfishPlugin.Logger.LogInfo("[UI] Saved last exit button path: " + path);
                }
            }
            catch (Exception ex)
            {
                PolyfishPlugin.Logger.LogWarning("[UI] Failed: " + ex.Message);
            }
        }
    }

    [HarmonyPatch]
    public class GameHooks
    {
        [HarmonyPostfix]
        [HarmonyPatch(typeof(LevelManager), "Start")]
        public static void OnStartGame()
        {
            GameState gameState = GameManager.GameState!;
            ReplayInterface replayInterface = UnityEngine.Object.FindObjectOfType<ReplayInterface>();

            if (replayInterface != null)
            {
                PolyfishPlugin.Logger!.LogInfo($"[Automation] Replay {gameState.Version} detected, initiating capture sequence..");
                
                replayInterface.timeline.Pause();

                Task.Run(async () =>
                {
                    // Wait for the timeline to be initialized
                    while (!replayInterface.timeline.isInitialized) 
                    {
                        await Task.Delay(50);
                    }

                    try
                    {
                        if (gameState.Version < 105)
                        {
                            PolyfishPlugin.Logger.LogWarning($"[Automation] Skipping outdated replay version: {gameState.Version} (Threshold: 105)");
                            
                            // remove the replay from the list
                            if (PolyfishPlugin.CurrentReplayUUID != null)
                            {
                                try 
                                {
                                    string fails_path = "/home/henry/Desktop/Coding/PolyAI/polyfish-scraper/data/replays_all.txt";
                                    if (File.Exists(fails_path))
                                    {
                                        var lines = new List<string>(File.ReadAllLines(fails_path));
                                        lines.RemoveAll(l => l.Contains(PolyfishPlugin.CurrentReplayUUID));
                                        File.WriteAllLines(fails_path, lines.ToArray());
                                        PolyfishPlugin.Logger.LogInfo($"[ReplaySequencer] Removed outdated {PolyfishPlugin.CurrentReplayUUID} from replays_all.txt");
                                    }
                                } 
                                catch (Exception ex)
                                {
                                    PolyfishPlugin.Logger.LogError($"[ReplaySequencer] Failed to remove {PolyfishPlugin.CurrentReplayUUID} from replays_all.txt: {ex.Message}");
                                }
                            }

                            goto ExitSequence;
                        }

                        // Safely capture initial game state before execution mutates it
                        PolyfishPlugin.InitialGameStateJson = PolyfishSerializer.SerializeGameState(gameState);
                        PolyfishPlugin.TargetExplorerCount = 0;
                        PolyfishPlugin.DiffedExplorerCount = 0;
                        PolyfishPlugin.PendingExplorer = null;
                        PolyfishPlugin.ExplorerRevealedTiles.Clear();
                        PolyfishPlugin.RuinsRewardCache.Clear();
                        PolyfishPlugin.RuinsTechCache.Clear();

                        // Count the total expected explorer commands
                        PolyfishPlugin.ExplorerIndices.Clear();
                        PolyfishPlugin.CommandByIndex.Clear();
                        // Sort turns to ensure indices are collected in order
                        var sortedTurns = new List<int>();
                        foreach (var key in replayInterface.timeline.timelineDataCache.Keys) sortedTurns.Add(key);
                        sortedTurns.Sort();

                        foreach (var turnKey in sortedTurns)
                        {
                            var turnData = replayInterface.timeline.timelineDataCache[turnKey];
                            foreach (var playerCommands in turnData.playerCommands)
                            {
                                foreach (var cmd in playerCommands.Value)
                                {
                                    PolyfishPlugin.CommandByIndex[cmd.index] = cmd.command!;
                                    
                                    if (cmd.command != null && PolyfishSerializer.IsExplorerCommand(cmd.command, gameState, true))
                                    {
                                        PolyfishPlugin.ExplorerIndices.Add(cmd.index);
                                    }
                                }
                            }
                        }
                        
                        // Sort ExplorerIndices just in case playerCommands iteration order was weird
                        PolyfishPlugin.ExplorerIndices.Sort();
                        PolyfishPlugin.TargetExplorerCount = PolyfishPlugin.ExplorerIndices.Count;

                        PolyfishPlugin.Logger.LogInfo($"[Automation] Found {PolyfishPlugin.TargetExplorerCount} potential explorers in timeline.");

                        replayInterface.timeline.Play();

                        // Start loop until replay is fully extracted
                        await TimelineHotkeys.SkipToNextExplorer();

                        PolyfishPlugin.RunOnMainThread(() =>
                        {
                            PolyfishPlugin.Logger.LogInfo("[Automation] Saving to db.");
                            PolyfishPlugin.API!.SaveReplaySync(PolyfishPlugin.InitialGameStateJson, replayInterface);
                        });

                        await Task.Delay(1000);

                        ExitSequence:
                        PolyfishPlugin.Logger.LogInfo("[Automation] Initiating exit sequence.");
                        // Resilient exit loop: keep clicking exit-like buttons until ReplayInterface is gone
                        var clickedPaths = new HashSet<string>();
                        for (int exitAttempt = 0; exitAttempt < 10; exitAttempt++)
                        {
                            // Check if we've already exited
                            bool replayGone = false;
                            PolyfishPlugin.RunOnMainThread(() =>
                            {
                                replayGone = !replayInterface || !replayInterface.gameObject;
                            });
                            await Task.Delay(500);
                            if (replayGone)
                            {
                                PolyfishPlugin.Logger.LogInfo("[Automation] Exit successful.");
                                break;
                            }

                            PolyfishPlugin.RunOnMainThread(() =>
                            {
                                // Search for an exit-like button we haven't clicked yet
                                // Fallback: if nothing new found, retry already-clicked buttons
                                var btn = FindExitLikeButton(clickedPaths) ?? FindExitLikeButton();
                                if (btn != null && btn.gameObject.activeInHierarchy)
                                {
                                    string path = PolyfishPlugin.GetGameObjectPath(btn.gameObject);
                                    string text = btn.GetComponentInChildren<TextMeshProUGUI>()?.text ?? "N/A";
                                    PolyfishPlugin.Logger.LogInfo($"[Automation] Exit attempt {exitAttempt + 1}: clicking '{text}' at {path}");
                                    clickedPaths.Add(path);
                                    ClickButton(btn);
                                }
                                else
                                {
                                    PolyfishPlugin.Logger.LogWarning($"[Automation] Exit attempt {exitAttempt + 1}: no exit-like button found, will retry...");
                                }
                            });

                            await Task.Delay(2000); // Wait for UI transitions
                        }
                    }
                    catch (Exception ex)
                    {
                        PolyfishPlugin.Logger.LogError("[Automation] Critical error: " + ex.Message + "\n" + ex.StackTrace);
                    }
                });
            }
        }

        private static readonly string[] ExitButtonTexts = new[]
        {
            "Exit", "Quit", "QUIT", "Finish Game", "Close", "Leave", "Back", "Done"
        };

        public static UIButtonBase? FindExitLikeButton(HashSet<string>? excludePaths = null)
        {
            foreach (var btn in UnityEngine.Object.FindObjectsOfType<UIButtonBase>())
            {
                if (!btn.gameObject.activeInHierarchy) continue;
                
                if (excludePaths != null && excludePaths.Contains(PolyfishPlugin.GetGameObjectPath(btn.gameObject)))
                {
                    continue;
                }
                
                var text = btn.GetComponentInChildren<TextMeshProUGUI>()?.text ?? "";
                foreach (var exitText in ExitButtonTexts)
                {
                    if (text.Equals(exitText, StringComparison.OrdinalIgnoreCase))
                    {
                        return btn;
                    }
                }
            }
            return null;
        }

        public static void ClickButton(UIButtonBase btn)
        {
            var eventData = new PointerEventData(EventSystem.current)
            {
                button = PointerEventData.InputButton.Left
            };

            EventSystem.current?.SetSelectedGameObject(btn.gameObject);
            ExecuteEvents.Execute(btn.gameObject, eventData, ExecuteEvents.pointerDownHandler);
            ExecuteEvents.Execute(btn.gameObject, eventData, ExecuteEvents.pointerUpHandler);
            ExecuteEvents.Execute(btn.gameObject, eventData, ExecuteEvents.pointerClickHandler);
            ExecuteEvents.Execute(btn.gameObject, eventData, ExecuteEvents.submitHandler);
        }
    }

    public enum ReplaySequencerState
    {
        Idle,
        Opening,
        CheckingDB,
        WaitingForLoad,
        Extracting,
        Closing
    }

    public class ReplaySequencer
    {
        private readonly Queue<string> _replayQueue = new();
        public ReplaySequencerState State { get; private set; } = ReplaySequencerState.Idle;
        private float _stateTimer = 0f;
        private bool _isCheckingDB = false;

        public ReplaySequencer()
        {
            LoadReplaysFromFile("/tmp/polyfish_replays.txt");
            
            // Scraper data sources
            string scraperDataPath = "/home/henry/Desktop/Coding/PolyAI/polyfish-scraper/data/";
            if (Directory.Exists(scraperDataPath))
            {
                foreach (var file in Directory.GetFiles(scraperDataPath, "replays_*.txt"))
                {
                    LoadReplaysFromFile(file);
                }
            }

            PolyfishPlugin.Logger.LogInfo($"[ReplaySequencer] Total loaded replays: {_replayQueue.Count}");
            if (_replayQueue.Count > 0)
            {
                State = ReplaySequencerState.Opening;
            }
            // GameManager.GameState.Map.tiles[0].improvement.type
        }

        private void LoadReplaysFromFile(string path)
        {
            if (File.Exists(path))
            {
                var lines = File.ReadAllLines(path);
                foreach (var line in lines)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        var id = line.Trim();
                        // Handle both full URLs and raw IDs
                        if (id.Contains("share.polytopia.io/"))
                        {
                            id = id[(id.LastIndexOf('/') + 1)..];
                        }
                        if (id.Contains("?id="))
                        {
                            id = id[(id.IndexOf("?id=") + 4)..];
                        }
                        
                        // Basic UUID/ID validation to avoid junk
                        if (id.Length > 5) 
                        {
                            _replayQueue.Enqueue(id);
                        }
                    }
                }
            }
        }

        public void Update()
        {
            if (State == ReplaySequencerState.Idle) return;

            _stateTimer += UnityEngine.Time.unscaledDeltaTime;

            switch (State)
            {
                case ReplaySequencerState.Opening:
                    if (_replayQueue.Count == 0)
                    {
                        State = ReplaySequencerState.Idle;
                        PolyfishPlugin.Logger.LogInfo("[ReplaySequencer] Queue empty. Finished automating replays.");
                        PolyfishPlugin.CurrentReplayUUID = null;
                        try { File.WriteAllText("/tmp/polyfish_replays.txt", ""); } catch {}
                        break;
                    }
                    State = ReplaySequencerState.CheckingDB;
                    _isCheckingDB = false;
                    break;
                
                case ReplaySequencerState.CheckingDB:
                    if (!_isCheckingDB)
                    {
                        _isCheckingDB = true;
                        var uuid = _replayQueue.Peek();
                        PolyfishPlugin.CurrentReplayUUID = uuid;
                        
                        Task.Run(async () => {
                            bool shouldProceed = await PolyfishPlugin.API!.CheckReplayExistsByUUIDAsync(uuid);
                            PolyfishPlugin.RunOnMainThread(() => {
                                if (!shouldProceed)
                                {
                                    PolyfishPlugin.Logger.LogInfo($"[ReplaySequencer] Replay {uuid} already exists in DB. Skipping.");
                                    _replayQueue.Dequeue();
                                    State = ReplaySequencerState.Opening;

                                    // remove from replays_all.txt
                                    Task.Run(() => {
                                        try 
                                        {
                                            string fails_path = "/home/henry/Desktop/Coding/PolyAI/polyfish-scraper/data/replays_all.txt";
                                            if (File.Exists(fails_path))
                                            {
                                                var lines = new List<string>(File.ReadAllLines(fails_path));
                                                lines.RemoveAll(l => l.Contains(uuid));
                                                File.WriteAllLines(fails_path, lines.ToArray());
                                                PolyfishPlugin.Logger.LogInfo($"[ReplaySequencer] Removed {uuid} from replays_all.txt");
                                            }
                                        } 
                                        catch (Exception ex)
                                        {
                                            PolyfishPlugin.Logger.LogError($"[ReplaySequencer] Failed to remove {uuid} from replays_all.txt: {ex.Message}");
                                        }
                                    });
                                }
                                else
                                {
                                    PolyfishPlugin.Logger.LogInfo($"[ReplaySequencer] Processing replay via DeepLinkManager: {uuid}");
                                    var uri = new Il2CppSystem.Uri($"steam://opengame:80/?id={uuid}");
                                    GameManager.GetDeepLinkManager().ProcessAsync(uri);
                                    // GameManager.GetDeepLinkManager().
                                    State = ReplaySequencerState.WaitingForLoad;
                                    _stateTimer = 0f;
                                }
                            });
                        });
                    }
                    break;

                case ReplaySequencerState.WaitingForLoad:
                    HandleFailedPopup();
                    
                    ReplayInterface replayInterface = UnityEngine.Object.FindObjectOfType<ReplayInterface>();
                    if (replayInterface != null && replayInterface.timeline != null)
                    {
                        State = ReplaySequencerState.Extracting;
                        _stateTimer = 0f;
                        PolyfishPlugin.Logger.LogInfo($"[ReplaySequencer] Replay logic detected! Going to extracting state.");
                    }
                    else if (_stateTimer > 10f)
                    {
                        PolyfishPlugin.Logger.LogWarning("[ReplaySequencer] Replay load timeout. Moving back to opening state.");
                        HandleFailedPopup(); 
                        if (State == ReplaySequencerState.WaitingForLoad)
                        {
                            _replayQueue.Dequeue();
                            State = ReplaySequencerState.Opening;
                            _stateTimer = 0f;
                        }
                    }
                    else if (_stateTimer > 5f)
                    {
                        HandleFailedPopup();
                    }
                    break;

                case ReplaySequencerState.Extracting:
                    if (UnityEngine.Object.FindObjectOfType<ReplayInterface>() == null && _stateTimer > 10f)
                    {
                        if (_stateTimer > 14f) 
                        {
                            PolyfishPlugin.Logger.LogInfo($"[ReplaySequencer] Extraction finished (ReplayInterface destroyed). Moving to next replay.");
                            _replayQueue.Dequeue();
                            State = ReplaySequencerState.Opening;
                            _stateTimer = 0f;
                        }
                    }
                    break;
                    
                case ReplaySequencerState.Closing:
                    break;
            }
        }

        private void HandleFailedPopup()
        {
            var btn = PolyfishPlugin.FindButtonByPath("Canvas/PopupManager/BasicPopup(Clone)/OverlayContainer/ButtonContainer/PopupButton_OK");
            
            // Backup Search. Sometimes the path doesn't quite match due to (Clone) placement
            if (btn == null)
            {
                foreach (var b in Resources.FindObjectsOfTypeAll<UIButtonBase>())
                {
                    if (b.gameObject.activeInHierarchy)
                    {
                        var text = b.GetComponentInChildren<TextMeshProUGUI>()?.text ?? "";
                        if (text.Equals("OK", StringComparison.OrdinalIgnoreCase))
                        {
                            var path = PolyfishPlugin.GetGameObjectPath(b.gameObject);
                            if (path.Contains("PopupManager"))
                            {
                                btn = b;
                                break;
                            }
                        }
                    }
                }
            }

            if (btn != null && btn.gameObject.activeInHierarchy)
            {
                PolyfishPlugin.Logger.LogWarning($"[ReplaySequencer] Clicked exit button.");
                
                if (!string.IsNullOrEmpty(PolyfishPlugin.CurrentReplayUUID))
                {
                    Task.Run(() => {
                        try 
                        {
                            string fails_path = "/home/henry/Desktop/Coding/PolyAI/polyfish-scraper/data/replays_all.txt";
                            if (File.Exists(fails_path))
                            {
                                var lines = new List<string>(File.ReadAllLines(fails_path));
                                lines.RemoveAll(l => l.Contains(PolyfishPlugin.CurrentReplayUUID));
                                File.WriteAllLines(fails_path, lines.ToArray());
                                PolyfishPlugin.Logger.LogInfo($"[ReplaySequencer] Removed {PolyfishPlugin.CurrentReplayUUID} from replays_all.txt");
                            }
                        } 
                        catch (Exception ex) 
                        {
                            PolyfishPlugin.Logger.LogWarning("Failed to remove from replays_all: " + ex.Message);
                        }
                    });
                }
                
                var eventData = new PointerEventData(EventSystem.current)
                {
                    button = PointerEventData.InputButton.Left
                };

                EventSystem.current?.SetSelectedGameObject(btn.gameObject);
                ExecuteEvents.Execute(btn.gameObject, eventData, ExecuteEvents.pointerDownHandler);
                ExecuteEvents.Execute(btn.gameObject, eventData, ExecuteEvents.pointerUpHandler);
                ExecuteEvents.Execute(btn.gameObject, eventData, ExecuteEvents.pointerClickHandler);
                ExecuteEvents.Execute(btn.gameObject, eventData, ExecuteEvents.submitHandler);

                if (_replayQueue.Count > 0)
                {
                    _replayQueue.Dequeue();
                }
                State = ReplaySequencerState.Opening;
                _stateTimer = 0f;
            }
        }
    }

    [HarmonyPatch]
    public class TimelineHotkeys
    {
        public static async Task SkipToNextExplorer()
        {
            while (true)
            {
                ReplayInterface replayInterface = UnityEngine.Object.FindObjectOfType<ReplayInterface>();
                if (replayInterface == null || replayInterface.timeline == null || !replayInterface.timeline.isInitialized)
                {
                    break;
                }

                var timeline = replayInterface.timeline;
                int currentIndex = timeline.currentIndex;

                int nextExplorerIndex = -1;
                int nextExplorerIdxInList = PolyfishPlugin.DiffedExplorerCount;
                
                if (nextExplorerIdxInList < PolyfishPlugin.ExplorerIndices.Count)
                {
                    nextExplorerIndex = PolyfishPlugin.ExplorerIndices[nextExplorerIdxInList];
                    
                    // If we are already sitting at or past this explorer (manually or accidentally), find the next one
                    while (nextExplorerIndex <= currentIndex && nextExplorerIdxInList + 1 < PolyfishPlugin.ExplorerIndices.Count)
                    {
                        nextExplorerIdxInList++;
                        nextExplorerIndex = PolyfishPlugin.ExplorerIndices[nextExplorerIdxInList];
                    }
                }

                if (nextExplorerIndex == -1)
                {
                    PolyfishPlugin.Logger.LogInfo($"[Automation] No more potential explorers found in the timeline.");
                    break;
                }

                PolyfishPlugin.RunOnMainThread(() => {
                    PolyfishPlugin.Logger.LogInfo($"[Automation] Jumping to potential explorer {nextExplorerIndex}");
                    timeline.SimulateToCommand(nextExplorerIndex);
                    replayInterface.timeline.Pause();
                });

                await Task.Delay(400);
                replayInterface.timeline.Play();
                await Task.Delay(400);
                replayInterface.timeline.Pause();
                await Task.Delay(400);

                // 2. Capture Snapshot
                HashSet<int> snapshot = new();
                List<int> techSnapshot = new();
                CommandBase? targetCmd = null;
                bool lookupDone = false;

                if (PolyfishPlugin.CommandByIndex.TryGetValue(nextExplorerIndex, out targetCmd))
                {
                    PolyfishPlugin.RunOnMainThread(() => {
                        snapshot = PolyfishSerializer.GetExploredTiles(GameManager.GameState, targetCmd.PlayerId);
                        techSnapshot = PolyfishSerializer.GetTribeTech(GameManager.GameState, targetCmd.PlayerId);
                        lookupDone = true;
                    });
                    
                    while (!lookupDone) await Task.Delay(50);

                    // Before jumping, capture the state, cause it might changed later
                    var typeId = targetCmd.Id?.ToLowerInvariant() ?? "";
                    if (typeId == "examineruins")
                    {
                        var ruinsCmd = targetCmd.Cast<ExamineRuinsCommand>();
                        var reward = ruinsCmd.GetRuinsRewardV119(
                            GameManager.GameState, 
                            targetCmd.PlayerId, 
                            GameManager.GameState.Map.tiles[PolyfishSerializer.GetIdx(ruinsCmd.Coordinates, GameManager.GameState.Settings.MapSize)]
                        );

                        PolyfishPlugin.RuinsRewardCache[targetCmd] = (int)reward;
                        
                        if (reward == RuinsReward.Explorer)
                        {
                            PolyfishPlugin.Logger.LogInfo($"[Automation] Explorer Ruin detected");
                        }
                    }
                }
                else
                {
                    PolyfishPlugin.Logger.LogWarning($"[Automation] Failed to find target on {nextExplorerIndex}");
                    break;
                }

                // 3. Jump to explorerIndex
                PolyfishPlugin.RunOnMainThread(() => {
                    PolyfishPlugin.Logger.LogInfo($"[Automation] Fast forwarding");
                    timeline.SimulateToCommand(nextExplorerIndex + 2);
                    replayInterface.timeline.Pause();
                });
                
                await Task.Delay(400);
                replayInterface.timeline.Play();
                await Task.Delay(400);
                replayInterface.timeline.Pause();
                await Task.Delay(400);

                // 4. Capture Final and Diff
                bool diffDone = false;
                PolyfishPlugin.RunOnMainThread(() => {
                    if (GameManager.GameState == null || GameManager.GameState.Map == null || GameManager.GameState.Settings == null)
                    {
                        PolyfishPlugin.Logger.LogError($"[Automation] Something went wrong. GameState is null!");
                        diffDone = true;
                        return;
                    }

                    replayInterface.OnForceRefreshHud();

                    PolyfishPlugin.Logger.LogInfo($"[Automation] TECH NOW: {string.Join(", ", PolyfishSerializer.GetTribeTech(GameManager.GameState, targetCmd.PlayerId))}");

                    var typeId = targetCmd.Id?.ToLowerInvariant() ?? "";
                   
                    // RE-VERIFY: Is it still an explorer now that we are here?
                    if (PolyfishSerializer.IsExplorerCommand(targetCmd, GameManager.GameState))
                    {
                        var currentTiles = PolyfishSerializer.GetExploredTiles(GameManager.GameState, targetCmd.PlayerId);
                        var revealing = currentTiles.Except(snapshot).ToList();
                        PolyfishPlugin.ExplorerRevealedTiles[targetCmd] = revealing;
                        PolyfishPlugin.Logger.LogInfo($"[Automation] Mapped explorer tiles {revealing.Count} [{PolyfishPlugin.DiffedExplorerCount + 1}/{PolyfishPlugin.TargetExplorerCount}]");
                    }
                    else
                    {
                        var ruinsCmd = targetCmd.Cast<ExamineRuinsCommand>();

                        ExamineRuinsAction action = new(
                            targetCmd.PlayerId, 
                            ruinsCmd.GetRuinsRewardV119(
                                GameManager.GameState,
                                targetCmd.PlayerId,
                                GameManager.GameState.Map.tiles[PolyfishSerializer.GetIdx(ruinsCmd.Coordinates, GameManager.GameState.Settings.MapSize)]
                            ),
                            ruinsCmd.Coordinates
                        );

                        // This is capturing Diplomacy moves?!!?
                        foreach (var item in GameManager.GameState.ActionStack)
                        {
                            if (item == null) continue;
                            PolyfishPlugin.Logger.LogInfo($"[Automation] Action: {item.GetActionType()}");
                        }

                        PolyfishPlugin.Logger.LogInfo($"[Automation] Ruin did not grant explorer or tech [{PolyfishPlugin.DiffedExplorerCount + 1}/{PolyfishPlugin.TargetExplorerCount}]");
                    }

                    PolyfishPlugin.DiffedExplorerCount++;
                    diffDone = true;
                });
                
                while (!diffDone) await Task.Delay(50);
            }
        }
    }
}
