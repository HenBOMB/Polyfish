
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Reflection;
using PolyMod.Api;
using UnityEngine;
using HarmonyLib;
using Il2CppInterop.Runtime.InteropTypes;
using Polytopia.Data;


namespace PolyfishAI.src
{
    public static class PolyfishSerializer
    {
        public static string SerializeGameState(GameState gs)
        {
            return JsonSerializer.Serialize(ExtractGameState(gs));
        }
        
        public static string SerializeReplay(string initialGameStateJson, ReplayInterface ri)
        {
            try
            {
                if (ri == null || ri.timeline == null)
                    return "{\"error\": \"ReplayInterface or timeline is null\"}";

                var timeline = ri.timeline;
                var turnsCache = timeline.timelineDataCache;
                // Since MapSize requires GameState and we already extracted it, we can parse it initially, 
                // but just getting it from GameManager is still fine if it matches. 
                int size = GameManager.GameState.Settings.MapSize;
                
                var turnsList = new List<object>();

                // Sort turns by key (turn number)
                var sortedTurnKeys = new List<int>();
                foreach (var key in turnsCache.Keys) sortedTurnKeys.Add(key);
                sortedTurnKeys.Sort();

                foreach (var turnNum in sortedTurnKeys)
                {
                    var turnData = turnsCache[turnNum];
                    
                    // Group all commands in this turn by their ACTUAL PlayerId
                    var groupedCommands = new Dictionary<int, List<object>>();
                    
                    foreach (var kvp in turnData.playerCommands)
                    {
                        var commandsList = kvp.Value;
                        for (int i = 0; i < commandsList.Count; i++)
                        {
                            var cmdData = commandsList[i];
                            if (cmdData != null && cmdData.command != null)
                            {
                                int actualPlayerId = (int)cmdData.command.PlayerId;
                                if (actualPlayerId == 255) continue; // Skip Nature commands

                                if (!groupedCommands.ContainsKey(actualPlayerId))
                                    groupedCommands[actualPlayerId] = new List<object>();
                                
                                var serializedCmd = (Dictionary<string, object>)SerializeCommand(cmdData.command, size);
                                if (PolyfishPlugin.ExplorerRevealedTiles.ContainsKey(cmdData.command))
                                {
                                    serializedCmd["_revealedTiles"] = PolyfishPlugin.ExplorerRevealedTiles[cmdData.command];
                                }
                                groupedCommands[actualPlayerId].Add(serializedCmd);
                            }
                        }
                    }

                    var playersList = new List<object>();
                    var sortedPlayerIds = groupedCommands.Keys.ToList();
                    sortedPlayerIds.Sort();

                    foreach (var playerId in sortedPlayerIds)
                    {
                        if (playerId == 255) continue;
                        playersList.Add(new
                        {
                            playerId = playerId,
                            commands = groupedCommands[playerId]
                        });
                    }

                    turnsList.Add(new
                    {
                        turn = turnNum,
                        players = playersList
                    });
                }

                var replayData = new Dictionary<string, object> { 
                    ["uuid"] = PolyfishPlugin.CurrentReplayUUID!,
                    ["turns"] = turnsList,
                    ["gameState"] = JsonSerializer.Deserialize<Dictionary<string, object>>(initialGameStateJson)!
                };

                return JsonSerializer.Serialize(replayData, new JsonSerializerOptions { WriteIndented = true });
            }
            catch (Exception ex)
            {
                PolyfishPlugin.Logger.LogError("[polyfish_ai] Replay serialization error: " + ex.Message + "\n" + ex.StackTrace);
                return "{\"error\": \"" + ex.Message + "\"}";
            }
        }

        public static Dictionary<string, object> ExtractGameState(GameState gs)
        {
            try
            {
                var settings = gs.Settings;
                var map = gs.Map;
                var size = settings.MapSize;
                var tiles = new Dictionary<int, object>();
                var structures = new Dictionary<int, object>();
                var resources = new Dictionary<int, object>();
                var _units = new Dictionary<int, List<object>>();
                var _cities = new Dictionary<int, List<object>>();

                var settingsDto = new Dictionary<string, object>
                {
                    ["version"] = gs.Version,
                    ["seed"] = gs.Seed,
                    ["turn"] = gs.CurrentTurn,
                    ["currentPlayerTurnId"] = gs.CurrentPlayerIndex,
                    ["size"] = settings.MapSize,
                    ["tileCount"] = settings.MapSize * settings.MapSize,
                    ["mapType"] = (int)settings.mapPreset,
                    ["mode"] = (int)settings.BaseGameMode,
                    ["gameName"] = settings.GameName,
                    ["winByCapital"] = settings.rules!.WinByCapital,
                    ["winByExtermination"] = settings.rules!.WinByExtermination,
                    ["_maxTribeCount"] = gs.PlayerStates.Count - 1, // - Nature
                };

                var root = new Dictionary<string, object>
                {
                    ["settings"] = settingsDto,
                    ["initialSeed"] = gs.Seed,
                };

                // 1. Pre-scan units to build ID-to-TribeIndex map (matches Rust's post_load mapping)
                var unitIdToIndex = new Dictionary<uint, int>();
                var tribeCounters = new Dictionary<int, int>();
                foreach (var tile in map.Tiles)
                {
                    var u = tile.unit;
                    if (u != null)
                    {
                        int owner = (int)u.owner;
                        if (!tribeCounters.ContainsKey(owner)) tribeCounters[owner] = 0;
                        unitIdToIndex[u.id] = tribeCounters[owner]++;
                    }
                }

                foreach (var tile in map.Tiles)
                {
                    int idx = GetIdx(tile.coordinates, size);

                    var explorerList = new List<int>();
                    foreach (var e in tile.explorers) explorerList.Add((int)e);

                    var effectsList = new List<int>();
                    foreach (var e in tile.effects) effectsList.Add((int)e);

                    var t = new Dictionary<string, object?>
                    {
                        ["coords"] = new { x = tile.coordinates.X, y = tile.coordinates.Y, idx },
                        ["rulingCityCoords"] = tile.rulingCityCoordinates.IsValid(size, size) ? new { x = tile.rulingCityCoordinates.X, y = tile.rulingCityCoordinates.Y, idx = GetIdx(tile.rulingCityCoordinates, size) } : null,
                        ["type"] = (int)tile.terrain,
                        ["explorers"] = explorerList,
                        ["hasRoad"] = tile.hasRoad,
                        ["hasRoute"] = tile.hasRoute,
                        ["capitalOf"] = (int)tile.capitalOf,
                        ["skinType"] = (int)tile.Skin,
                        ["climate"] = (int)tile.climate,
                        ["owner"] = (int)tile.owner,
                        ["effects"] = effectsList,
                    };

                    var imp = tile.improvement;
                    if (imp != null)
                    {
                        structures[idx] = new
                        {
                            type = (int)imp.type,
                            level = (int)imp.level,
                            founded = (int)imp.founded,
                        };

                        // Village & Owned = City
                        if ((int)imp.type == 1 && (int)tile.owner > 0)
                        {
                            if (!_cities.ContainsKey((int)tile.owner)) _cities[(int)tile.owner] = new List<object>();

                            var rewardsList = new List<int>();
                            foreach (var r in imp.rewards) rewardsList.Add((int)r);

                            _cities[(int)tile.owner].Add(new
                            {
                                owner = (int)tile.owner,
                                name = imp.name,
                                idx = idx,
                                population = (int)imp.population,
                                progress = (int)imp.xp,
                                borderSize = (int)imp.borderSize,
                                connectedToCapital = imp.connectedToCapitalOfPlayer == 1,
                                level = (int)imp.level,
                                production = (int)imp.production,
                                rewards = rewardsList,
                            });
                        }
                    }

                    var res = tile.resource;
                    if (res != null)
                    {
                        resources[idx] = new { type = (int)res.type };
                    }

                    var unit = tile.unit;
                    if (unit != null)
                    {
                        if (!_units.ContainsKey((int)unit.owner)) _units[(int)unit.owner] = new List<object>();

                        var unitEffects = new List<int>();
                        foreach (var e in unit.effects) unitEffects.Add((int)e);

                        var uDto = new Dictionary<string, object?>
                        {
                            ["owner"] = (int)unit.owner,
                            ["type"] = (int)unit.type,
                            ["hp"] = (int)Math.Round((float)unit.health / 10f),
                            ["promoted"] = unit.promotionLevel != 0,
                            ["xp"] = unit.xp,
                            ["coords"] = new { x = unit.coordinates.X, y = unit.coordinates.Y, idx = idx },
                            ["prevCoords"] = unit.previousTurnEndCoordinates.IsValid(size, size) ? new { x = unit.previousTurnEndCoordinates.X, y = unit.previousTurnEndCoordinates.Y, idx = GetIdx(unit.previousTurnEndCoordinates, size) } : new { x = -1, y = -1, idx = -1 },
                            ["homeCoords"] = unit.home.IsValid(size, size) ? new { x = unit.home.X, y = unit.home.Y, idx = GetIdx(unit.home, size) } : null,
                            ["flipped"] = unit.flipped,
                            ["direction"] = unit.direction,
                            ["createdTurn"] = unit.createdTurn,
                            ["passengerType"] = unit.passengerUnit != null ? (int)unit.passengerUnit.type : null,
                            ["moved"] = unit.moved,
                            ["attacked"] = unit.attacked,
                            ["effects"] = unitEffects,
                            ["converted"] = unit.birthClimate != gs.PlayerStates[unit.owner].climate,
                            ["attacksPerformed"] = 0,
                            ["parentUnitIdx"] = unit.HasLeader() && unitIdToIndex.TryGetValue(unit.leader, out int pIdx) ? (int?)pIdx : null,
                            ["childUnitIdx"] = unit.HasFollower() && unitIdToIndex.TryGetValue(unit.follower, out int cIdx) ? (int?)cIdx : null
                        };
                        _units[(int)unit.owner].Add(uDto);
                    }

                    tiles[idx] = t;
                }

                var tribes = new Dictionary<int, object>();

                foreach (var p in gs.PlayerStates)
                {
                    if ((int)p.Id == 255) continue; // Skip Nature

                    var tech = new List<int>();
                    foreach (var techType in p.availableTech) tech.Add((int)techType);

                    var builtImps = new List<int>();
                    if (p.builtUniqueImprovements != null)
                        foreach (var impType in p.builtUniqueImprovements) builtImps.Add((int)impType);

                    var known = new List<int>();
                    if (p.knownPlayers != null)
                        foreach (var kp in p.knownPlayers) known.Add((int)kp);

                    var relations = new Dictionary<string, object>();
                    if (p.relations != null)
                    {
                        foreach (var r in p.relations)
                        {
                            if ((int)r.Key == 255) continue;
                            relations[r.Key.ToString()] = new
                            {
                                playerId = (int)r.Key,
                                state = (int)r.Value.State,
                                lastAttackTurn = r.Value.LastAttackTurn,
                                embassyLevel = r.Value.EmbassyLevel,
                                lastPeaceBrokenTurn = r.Value.LastPeaceBrokenTurn,
                                firstMeet = r.Value.FirstMeet,
                                embassyBuildTurn = r.Value.EmbassyBuildTurn,
                                previousAttackTurn = r.Value.PreviousAttackTurn
                            };
                        }
                    }

                    tribes[(int)p.Id] = new
                    {
                        id = (int)p.Id,
                        username = p.UserName,
                        builtUniqueImprovements = builtImps,
                        knownPlayers = known,
                        bot = p.AutoPlay,
                        score = p.score,
                        stars = p.Currency,
                        type = (int)p.tribe,
                        killerId = (int)p.killerId,
                        killedTurn = p.killedTurn,
                        resignedTurn = p.resignedTurn,
                        startingTileCoords = new { x = p.startTile.X, y = p.startTile.Y, idx = GetIdx(p.startTile, size) },
                        kills = p.kills,
                        casualties = p.casualities,
                        tech_vanilla = tech,
                        relations = relations,
                        cities = _cities.ContainsKey((int)p.Id) ? _cities[(int)p.Id] : new List<object>(),
                        units = _units.ContainsKey((int)p.Id) ? _units[(int)p.Id] : new List<object>(),

                        attacked_this_turn = _units.ContainsKey((int)p.Id) && _units[(int)p.Id].Any(u => (bool)((Dictionary<string, object>)u)["attacked"]),
                        pacifist_turns = 0,
                        conversions = 0,
                    };
                }

                root["tiles"] = tiles;
                root["tribes"] = tribes;
                root["resources"] = resources;
                root["structures"] = structures;

                return root;
            }
            catch (Exception ex)
            {
                PolyfishPlugin.Logger.LogError($"Serialization error: {ex.Message}\n{ex.StackTrace}");
                return new Dictionary<string, object> { ["error"] = "Replay serialization failed" };
            }
        }

        private static T? TryCast<T>(CommandBase cmd) where T : Il2CppObjectBase
        {
            if (cmd == null) return null;
            try { return cmd.Cast<T>(); } catch { return null; }
        }

        public static object SerializeCommand(CommandBase command, int size)
        {
            var typeId = command.Id;
            var result = new Dictionary<string, object>
            {
                ["moveType"] = -1 // Default: Invalid
            };

            string tid = typeId?.ToLowerInvariant() ?? "";

            switch (tid)
            {
                // ---- Step ---- //
                case "move":
                    var move = TryCast<MoveCommand>(command)!;
                    result["moveType"] = 1; // Step
                    result["src"] = GetIdx(move.From, size);
                    result["target"] = GetIdx(move.To, size);
                    break;

                // ---- Attack ---- //
                case "attack":
                    var attack = TryCast<AttackCommand>(command)!;
                    result["moveType"] = 2; // Attack
                    result["src"] = GetIdx(attack.Origin, size);
                    result["target"] = GetIdx(attack.Target, size);
                    break;

                // ---- Summon ---- //
                case "train":
                    var train = TryCast<TrainCommand>(command)!;
                    result["moveType"] = 4; // Summon
                    result["type"] = (int)train.Type;
                    result["src"] = GetIdx(train.Coordinates, size);
                    break;

                case "upgrade":
                    var upgrade = TryCast<UpgradeCommand>(command)!;
                    result["moveType"] = 4; // Upgrade is mapped to Summon (4) in simulator
                    result["type"] = (int)upgrade.Type;
                    result["src"] = GetIdx(upgrade.Coordinates, size);
                    break;

                // ---- Structure, Harvest, Abilities (Improvement) ---- //
                case "build":
                    var improv = TryCast<BuildCommand>(command)!;
                    
                    // Harvests
                    if (improv.Type == ImprovementData.Type.Fishing
                        || improv.Type == ImprovementData.Type.Hunting
                        || improv.Type == ImprovementData.Type.HarvestFruit
                        || improv.Type == ImprovementData.Type.HarvestSpores)
                    {
                        result["moveType"] = 5; // MoveType::Harvest
                        result["target"] = GetIdx(improv.Coordinates, size);
                    }
                    else if (improv.Type == ImprovementData.Type.WhaleHunting)
                    {
                        PolyfishPlugin.Logger.LogWarning("**POLYFISH BRIDGE**: Translating legacy WhaleHunting to Harvest.");
                        result["moveType"] = 5; // MoveType::Harvest
                        result["target"] = GetIdx(improv.Coordinates, size);
                    }
                    else if (improv.Type == ImprovementData.Type.BurnForest
                        || improv.Type == ImprovementData.Type.ClearForest
                        || improv.Type == ImprovementData.Type.GrowForest) 
                    {
                        result["moveType"] = 3; // Ability
                        result["type"] = improv.Type == ImprovementData.Type.BurnForest? 1 : improv.Type == ImprovementData.Type.ClearForest? 2 : 3; // AbilityType::ForestAction
                        result["target"] = GetIdx(improv.Coordinates, size);
                    }
                    else if (improv.Type == ImprovementData.Type.EnchantAnimal)
                    {
                        result["moveType"] = 3; // Ability
                        result["type"] = 23; // AbilityType::EnchantAnimal
                        result["target"] = GetIdx(improv.Coordinates, size);
                    }
                    else if (improv.Type == ImprovementData.Type.StarFishing)
                    {
                        result["moveType"] = 8; // MoveType::Capture
                        result["src"] = GetIdx(improv.Coordinates, size);
                    }
                    // TODO: Burn Spores?, Cultivate?
                    else 
                    {
                        // Default to Build (structures, roads, etc.)
                        result["moveType"] = 6; // Build
                        result["target"] = GetIdx(improv.Coordinates, size);
                        result["type"] = (int)improv.Type;
                    }
                    break;

                // ---- Technology ---- //
                case "research":
                    var research = TryCast<ResearchCommand>(command)!;
                    result["moveType"] = 7; // Research
                    result["type"] = (int)research.Type;
                    break;

                // ---- Capture ---- //
                case "capture":
                    var capture = TryCast<CaptureCommand>(command)!;
                    result["moveType"] = 8; // Capture
                    result["src"] = GetIdx(capture.Coordinates, size);
                    break;
                
                case "examineruins":
                    var examineRuins = TryCast<ExamineRuinsCommand>(command)!;
                    result["moveType"] = 8; // Capture
                    result["src"] = GetIdx(examineRuins.Coordinates, size);
                    
                    if (PolyfishPlugin.RuinsRewardCache.TryGetValue(command, out int cachedReward))
                    {
                        result["_reward"] = cachedReward;
                        // If it was a free tech reward, include the tech ID in _tech or type
                        if (cachedReward == 4 && PolyfishPlugin.RuinsTechCache.TryGetValue(command, out int techId))
                        {
                            result["_type"] = techId;
                        }
                    }
                    else
                    {
                        // Should never get hear but either way just in case..
                        PolyfishPlugin.Logger.LogWarning("[Automation] Ruins reward not found in cache, using fallback");
                        // Fallback to the current ruin's state on the tile.
                        var ruinsReward = (int)examineRuins.GetRuinsRewardV119(
                            GameManager.GameState, 
                            command.PlayerId, 
                            GameManager.GameState.Map.tiles[GetIdx(examineRuins.Coordinates, GameManager.GameState.Settings.MapSize)]
                        );
                        result["_reward"] = ruinsReward;
                    }
                    break;

                case "cityreward":
                    var reward = TryCast<CityRewardCommand>(command)!;
                    result["moveType"] = 9; // Reward
                    result["type"] = (int)reward.Reward;
                    result["target"] = GetIdx(reward.Coordinates, size);
                    break;

                case "destroy":
                    var destroy = TryCast<DestroyCommand>(command)!;
                    result["moveType"] = 3; // Ability
                    result["type"] = 4; // AbilityType.Destroy
                    result["target"] = GetIdx(destroy.Coordinates, size);
                    break;

                case "decompose":
                    var decompose = TryCast<DecomposeCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 5; // AbilityType.Decompose
                    result["target"] = GetIdx(decompose.Coordinates, size);
                    break;

                case "recover":
                    var recover = TryCast<RecoverCommand>(command)!;
                    result["moveType"] = 3; // Ability
                    result["type"] = 7; // AbilityType.Recover
                    result["src"] = GetIdx(recover.Coordinates, size);
                    break;

                case "disband":
                    var disband = TryCast<DisbandCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 8; // AbilityType.Disband
                    result["src"] = GetIdx(disband.Coordinates, size);
                    break;

                case "healothers":
                    var healOthers = TryCast<HealOthersCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 9; // AbilityType.HealOthers
                    result["src"] = GetIdx(healOthers.Coordinates, size);
                    break;

                case "freezearea":
                    var freezeArea = TryCast<FreezeAreaCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 13; // AbilityType.FreezeArea
                    result["src"] = GetIdx(freezeArea.Coordinates, size);
                    break;

                case "boost":
                    var boost = TryCast<BoostCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 14; // AbilityType.Boost
                    result["src"] = GetIdx(boost.Coordinates, size);
                    break;

                case "promote":
                    var promote = TryCast<PromoteCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 16; // AbilityType.Promote
                    result["src"] = GetIdx(promote.Coordinates, size);
                    break;

                case "explode":
                    var explode = TryCast<ExplodeCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 15; // AbilityType.Explode
                    result["src"] = GetIdx(explode.Coordinates, size);
                    break;

                case "breakice":
                    var breakIce = TryCast<BreakIceCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 17; // AbilityType.BreakIce
                    result["src"] = GetIdx(breakIce.Coordinates, size);
                    break;
        
                case "breakpeace":
                    var breakPeace = TryCast<BreakPeaceCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 18; // AbilityType.BreakPeace
                    result["src"] = (int)breakPeace.OpponentId;
                    break;

                case "peacerequestresponse":
                    var peaceRequestResponse = TryCast<PeaceRequestResponseCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 19; // AbilityType.PeaceRequestResponse
                    result["target"] = peaceRequestResponse.Accepted ? 1 : 0;
                    result["src"] = (int)peaceRequestResponse.OpponentId;
                    break;
                
                case "establishembassy":
                    var establishEmbassy = TryCast<EstablishEmbassyCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 20; // AbilityType.EstablishEmbassy
                    result["target"] = GetIdx(establishEmbassy.Coordinates, size);
                    result["src"] = (int)establishEmbassy.OpponentId;
                    break;

                case "peacetreaty":
                    var peaceTreaty = TryCast<PeaceTreatyCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 21; // AbilityType.PeaceTreaty
                    result["target"] = GetIdx(peaceTreaty.Coordinates, size);
                    result["src"] = (int)peaceTreaty.OpponentId;
                    break;

                case "destroyembassy":
                    var destroyEmbassy = TryCast<DestroyEmbassyCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 22; // AbilityType.DestroyEmbassy
                    result["target"] = GetIdx(destroyEmbassy.Coordinates, size);
                    result["src"] = (int)destroyEmbassy.OpponentId;
                    break;

                case "startmatch":
                case "endmatch":
                    result["moveType"] = -1;
                    break;

                case "endturn":
                    result["moveType"] = 10; // EndTurn
                    break;

                case "resign":
                    result["moveType"] = 11; // Resign
                    break;

                case "swarm":
                    var swarm = TryCast<SwarmCommand>(command)!;
                    result["moveType"] = 3;
                    result["type"] = 14; // SkillType::Swarm
                    result["src"] = GetIdx(swarm.Coordinates, size);
                    break;
                    
                default:
                    result["error"] = "Unknown command";
                    result["tid"] = tid;
                    PolyfishPlugin.Logger.LogWarning($"Unknown/Unhandled command type ID: {tid}");
                    break;
            }

            return result;
        }

        public static int GetIdx(WorldCoordinates coordsObj, int size)
        {
            var x = coordsObj.X;
            var y = coordsObj.Y;
            return y * size + x;
        }

        public static bool IsExplorerCommand(CommandBase command, GameState state, bool potential = false)
        {
            var typeId = command.Id?.ToLowerInvariant() ?? "";
            
            if (typeId == "cityreward")
            {
                return TryCast<CityRewardCommand>(command)!.Reward == CityReward.Explorer;
            }
            else if (typeId == "examineruins")
            {
                if (potential) return true; // During pre-scan, treat all ruins as potential explorers

                var ruinsCmd = TryCast<ExamineRuinsCommand>(command);
                if (ruinsCmd == null) return false;

                int reward;
                if (PolyfishPlugin.RuinsRewardCache.TryGetValue(command, out int cached))
                {
                    reward = cached;
                }
                else
                {
                    // Fallback to the current ruin's state on the tile.
                    reward = (int)ruinsCmd.GetRuinsRewardV119(
                        GameManager.GameState, 
                        command.PlayerId, 
                        GameManager.GameState.Map.tiles[GetIdx(ruinsCmd.Coordinates, state.Settings.MapSize)]
                    );
                }

                return (int)RuinsReward.Explorer == reward;
            }
            return false;
        }

        public static HashSet<int> GetExploredTiles(GameState state, int playerId)
        {
            var tiles = new HashSet<int>();
            if (state == null || state.Map == null || state.Map.Tiles == null) return tiles;
            
            int size = state.Settings.MapSize;
            foreach (var tile in state.Map.Tiles)
            {
                if (tile.explorers.Contains((byte)playerId))
                    tiles.Add(GetIdx(tile.coordinates, size));
            }
            return tiles;
        }

        public static List<int> GetTribeTech(GameState state, int playerId)
        {

            GameLogicData data = GameManager.GameState.GameLogicData;


            var techList = new List<int>();
            // var availableTech = state.PlayerStates[playerId].availableTech;
            var availableTech = data.GetUnlockedTech(GameManager.GameState.PlayerStates[playerId]);
            foreach (var techType in availableTech)
            {
                techList.Add((int)techType.type);
            }
            return techList;
        }
    }
}
