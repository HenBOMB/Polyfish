import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport

from starlette.applications import Starlette
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
from starlette.routing import Mount, Route

from pathlib import Path
import uvicorn, importlib
from datetime import datetime

### PROMPTS QUE SE REPITEN! ###

_MES = 'Numbero del mes'
_MAX_RESULTS = 'Cantidad de resultados a devolver'

### SETTINGS ###

endpoint = "/messages/"
app = Server("vicky-mcp")
port = 6789
host = "0.0.0.0"
tools = {}
sse = SseServerTransport(endpoint)

async def handle_sse(request):
    try:
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            init_options = app.create_initialization_options()
            await app.run(streams[0], streams[1], init_options)#, timeout=1000 * 15
            # await wait_for(
            # )
    except TimeoutError:
        print("Sesión SSE expirada por timeout")
    except Exception as e:
        print(f"Error en sesión SSE: {e}")
    print('close')
    return Response(status_code=204)
        
starlette_app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount(endpoint, app=sse.handle_post_message),
    ],
)

### MAIN ###

def load_tools():
    tools_dir = Path(__file__).parent / "tools"
    for tool_file in tools_dir.glob("*.py"):
        if tool_file.stem != "__init__":
            module_name = f"tools.{tool_file.stem}"
            module = importlib.import_module(module_name)
            tools[tool_file.stem] = module.Tool()

@app.call_tool()
async def fetch_tool(
    tool_name: str, arguments: dict
) -> types.TextContent:
    # Fetch tool from list_tools()
    # tool_info = next((tool for tool in await list_tools() if tool.name == tool_name), None)
    
    try:
        # Sanitize (optional inputs)
        checkstr = ['nombre', 'evento', 'nombre_corto', 'deporte', 'target', 'apellido', 'operador', 'group_by', 'tipo', 'federacion']
        for key in checkstr:
            arguments[key] = arguments[key] if key in arguments else ''

        checkint = [['max_results', 5], ['mes_nro', datetime.now().month]]
        for key, default in checkint:
            arguments[key] = arguments[key] if key in arguments else default

        output = tools[tool_name](arguments)

        return [types.TextContent(type="text", text=str(output))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"An error occurred: {str(e)}")]

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name='becados',
            description='Informacion sobre becas, becados y becarios',
            inputSchema={
                'type': 'object',
                'required': ['query'],
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Query tipo pandas para filtrar',
                    },
                    'operador': {
                        'type': 'string',
                        'description': 'Count, list, sum or mean',
                    },
                    'group_by': {
                        'type': 'string',
                        'description': 'Columnas a agrupar los resultados',
                    },
                    'target': {
                        'type': 'string',
                        'description': 'Valor del operador mean a usar',
                    },
                    'max_results': {
                        'type': 'number',
                        'description': _MAX_RESULTS,
                    }
                },
            }
        ),
        # autoridades
        types.Tool(
            name='autoridades',
            description='Trae informacion sobre autoridades del enard or nacionales, figuras destacadas en el ámbito deportivo argentino. Eg: Diojenes de Urquiza (Enard)',
            inputSchema={
                'type': 'object',
                'required': ['tipo'],
                'properties': {
                    'nombre': {
                        'type': 'string',
                        'description': 'Nombre de la autoridad',
                    },
                    'apellido': {
                        'type': 'string',
                        'description': 'Apellido de la autoridad',
                    },
                    'tipo': {
                        'type': 'string',
                        'description': '"Nacional "o "Enard"',
                    },
                },
            },
        ),
        # deportistas
        types.Tool(
            name='deportistas',
            description='Descripcion sobre los deportistas argentinos destacados. Eg: Lionel Messi, Scioli',
            inputSchema={
                'type': 'object',
                'required': [],
                'properties': {
                    'nombre': {
                        'type': 'string',
                        'description': 'Nombre del deportista',
                    },
                    'apellido': {
                        'type': 'string',
                        'description': 'Apellido del deportista',
                    },
                    'deporte': {
                        'type': 'string',
                        'description': 'Deporte que juega el deportista',
                    },
                    'max_results': {
                        'type': 'number',
                        'description': _MAX_RESULTS,
                    }
                },
            },
        ),
        types.Tool(
            name='eventos_deportivos',
            description='Informacion sobre los eventos deportivos en el ENARD del 2025',
            inputSchema={
                'type': 'object',
                'required': [],
                'properties': {
                    'evento': {
                        'type': 'string',
                        'description': 'Nombre del evento',
                    },
                    'deporte': {
                        'type': 'string',
                        'description': 'Nombre del deporte',
                    },
                    'mes_nro': {
                        'type': 'number',
                        'description': _MES,
                    },
                    'max_results': {
                        'type': 'number',
                        'description': _MAX_RESULTS,
                    },
                },
            },
        ),
        types.Tool(
            name='federaciones',
            description='Trae informacion sobre las federaciones',
            inputSchema={
                'type': 'object',
                'required': [],
                'properties': {
                    'federacion': {
                        'type': 'string',
                        'description': 'Federacion a buscar',
                    },
                    'deporte': {
                        'type': 'string',
                        'description': 'Deporte al que corresponde la federacion',
                    },
                    'max_results': {
                        'type': 'number',
                        'description': _MAX_RESULTS,
                    }
                },
            },
        ),
        types.Tool(
            name='selecciones_nacionales',
            description='Trae informacion sobre las selecciones nacionales de Argentina (Los murcielagos, Las leonas)',
            inputSchema={
                'type': 'object',
                'required': [],
                'properties': {
                    'nombre': {
                        'type': 'string',
                        'description': 'Nombre de la seleccion nacional',
                    },
                    'deporte': {
                        'type': 'string',
                        'description': 'Deporte al que corresponde la seleccion nacional',
                    },
                    'max_results': {
                        'type': 'number',
                        'description': _MAX_RESULTS,
                    }
                },
            },
        ),
        types.Tool(
            name='valoraciones',
            description='Aqui almacenas las valoraciones y comentarios de los usuarios a la charla que tuvieron',
            inputSchema={
                'type': 'object',
                'required': ['nombre', 'apellido', 'userid', 'chatid', 'valoracion', 'comentario'],
                'properties': {
                    'nombre': {
                        'type': 'string',
                        'description': 'Nombre del usuario',
                    },
                    'apellido': {
                        'type': 'string',
                        'description': 'Apellido del usuario',
                    },
                    'userid': {
                        'type': 'string',
                        'description': 'ID del usuario',
                    },
                    'chatid': {
                        'type': 'string',
                        'description': 'ID del chat',
                    },
                    'valoracion': {
                        'type': 'string',
                        'description': 'Valoracion del usuario',
                    },
                    'comentario': {
                        'type': 'string',
                        'description': 'Comentario del usuario',
                    },
                },
            },
        ),
    ]

load_tools()
uvicorn.run(starlette_app, host=host, port=port, workers=1, limit_concurrency=100, limit_max_requests=1000)
