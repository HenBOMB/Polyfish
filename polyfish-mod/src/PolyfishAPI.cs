using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using BepInEx.Logging;

namespace PolyfishAI.src
{
    public class PolyfishAPI
    {
        private readonly HttpClient _client;
        private readonly string _baseUrl;
        private readonly ManualLogSource _logger;

        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            ReferenceHandler = ReferenceHandler.Preserve,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            WriteIndented = false,
            MaxDepth = 1024,
        };

        static PolyfishAPI()
        {
            _jsonOptions.Converters.Add(new IntPtrConverter());
            _jsonOptions.Converters.Add(new UIntPtrConverter());
        }

        public PolyfishAPI(string baseUrl = "http://localhost:3000", ManualLogSource? logger = null)
        {
            _client = new HttpClient();
            _client.Timeout = TimeSpan.FromSeconds(10);
            _baseUrl = baseUrl.TrimEnd('/');
            _logger = logger ?? Logger.CreateLogSource("PolyfishBridge");

            _logger.LogInfo($"PolyfishBridge connected to {_baseUrl}");
        }

        /// <summary>
        /// Makes a request to the PolyfishAI server.
        /// </summary>
        private async Task<string?> SendRequestAsync(string edge, string payload, HttpMethod? method = null)
        {
            method ??= HttpMethod.Post;
            var url = $"{_baseUrl}/{edge.ToLower().TrimStart('/')}";

            try
            {
                var content = new StringContent(payload, Encoding.UTF8, "application/json");

                HttpRequestMessage request = new(method, url)
                {
                    Content = content
                };

                var response = await _client.SendAsync(request);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogWarning($"Request to {url} failed: {response.StatusCode} - {responseContent}");
                    return null;
                }

                return responseContent;
            }
            catch (Exception ex)
            {
                _logger.LogError($"Exception during bridge request to {url}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Makes a request to the PolyfishAI server.
        /// </summary>
        private async Task<string?> SendRequestAsync(string edge, object payload, HttpMethod? method = null)
        {
            return await SendRequestAsync(edge, JsonSerializer.Serialize(payload, _jsonOptions), method);
        }
       
        /// <summary>
        /// Sends the current GameState to the PolyfishAI server for saving.
        /// </summary>
        public async Task SaveStateAsync(GameState gameState)
        {
            try
            {
                _logger.LogInfo("Saving GameState...");
                await SendRequestAsync("save", PolyfishSerializer.SerializeGameState(gameState));
                _logger.LogInfo("Saved successfully!");
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to save game state: {ex.Message}");
            }
        }

        /// <summary>
        /// Sends the current replay to the PolyfishAI server for saving.
        /// </summary>
        public void SaveReplaySync(string initialGameStateJson, ReplayInterface replayInterface)
        {
            try
            {
                _ = SendRequestAsync("replay/save", PolyfishSerializer.SerializeReplay(initialGameStateJson, replayInterface));
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to save replay stat e: {ex.Message}");
            }
        }

        /// <summary>
        /// Checks if a replay with the given UUID already exists in the DB.
        /// Returns true if we should PROCEED (replay does NOT exist), false if we should SKIP.
        /// </summary>
        public async Task<bool> CheckReplayExistsByUUIDAsync(string uuid)
        {
            try
            {
                var payload = JsonSerializer.Serialize(new { uuid = uuid });
                var response = await SendRequestAsync("replay/check", payload);
                if (response != null)
                {
                    using var doc = JsonDocument.Parse(response);
                    if (doc.RootElement.TryGetProperty("proceed", out var proceedProp))
                    {
                        return proceedProp.GetBoolean();
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Failed to check replay existence by UUID: {ex.Message}");
            }
            // Default to proceeding if the check fails
            return true;
        }
    }

    public class IntPtrConverter : JsonConverter<IntPtr>
    {
        public override IntPtr Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) => IntPtr.Zero;
        public override void Write(Utf8JsonWriter writer, IntPtr value, JsonSerializerOptions options) => writer.WriteStringValue("0x" + value.ToString("X"));
    }

    public class UIntPtrConverter : JsonConverter<UIntPtr>
    {
        public override UIntPtr Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) => UIntPtr.Zero;
        public override void Write(Utf8JsonWriter writer, UIntPtr value, JsonSerializerOptions options) => writer.WriteStringValue("0x" + value.ToString("X"));
    }
}
