import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from coordinate import CoordinateTransformer, CoordinateGet


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有 origins
    allow_credentials=True,  # 允许携带凭证（如 cookies）
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 headers
)

converter = CoordinateTransformer()
coordiget = CoordinateGet()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    converter.load_data()
    try:
        # 创建读取和写入任务
        read_task = asyncio.create_task(read_from_websocket(websocket))
        write_task = asyncio.create_task(write_to_websocket(websocket))

        # 等待任务完成
        await asyncio.gather(read_task, write_task)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        converter.save_data()
        await websocket.close()


async def read_from_websocket(websocket: WebSocket):
    while True:
        json_coords = await websocket.receive_json()
        gcoords = json_coords["gamex"], json_coords["gamey"]
        mcoords = json_coords["map_x"], json_coords["map_y"]
        converter.set_data(gcoords, mcoords)
        converter.train_model()


async def write_to_websocket(websocket: WebSocket):
    while True:
        try:
            x, y = coordiget.get_coordinate()
            conv = converter.predict([x, y])
            print(x, y)
            coordi = {
                "gamex": float(x),
                "gamey": float(y),
                "map_x": float(conv[0]),
                "map_y": float(conv[1]),
            }

            # 将数据转换为标准的 Python 数据类型
            await websocket.send_json(coordi)
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"WebSocket error: {e}")
            await asyncio.sleep(10)


@app.post("/clean_data")
async def clean_data():
    converter.clean_data()
    return {"message": "All coordinate data has been cleared."}


if __name__ == "__main__":
    uvicorn.run(app="server:app", host="0.0.0.0", port=8001, reload=True)
