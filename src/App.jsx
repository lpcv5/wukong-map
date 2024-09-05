import { useRef, useEffect, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import heibox from "./assets/heibox.json";
import "./App.css";
import proj4 from "proj4";
import axios from 'axios';  // 确保你已经安装了 axios

mapboxgl.accessToken =
  "";
mapboxgl.baseApiUrl = "https://api.mapbox.cn";
const HEY_TILE_PATH = `/chapter1/{z}_{y}_{x}.jpg`;
function App() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const markerRef = useRef(null);
  const [ws, setWs] = useState(null);
  const [trainx, setTrainx] = useState(0.7);
  const [trainy, setTrainy] = useState(-0.7);
  const [gamex, setGamex] = useState(0);
  const [gamey, setGamey] = useState(0);
  const [mapx, setMapx] = useState(-0.7);
  const [mapy, setMapy] = useState(0.7);

  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:8001/ws");
    socket.onopen = () => {
      console.log("connected");
    };
    socket.onmessage = (event) => {
      const jsonData = JSON.parse(event.data);
      if (jsonData.map_x > -1.4 && jsonData.map_x < 0) {
        setMapx(jsonData.map_x);
        setMapy(jsonData.map_y);
      }
      setGamex(jsonData.gamex);
      setGamey(jsonData.gamey);
      console.log(jsonData);
    };
    socket.onerror = (error) => {
      console.log("error", error);
    };
    socket.onclose = (event) => {
      console.log("closed", event);
    };
    // 将WebSocket对象保存到状态中
    setWs(socket);

    // 组件卸载时关闭WebSocket连接
    return () => {
      socket.close();
    };
  }, []);

  useEffect(() => {
    if (ws) {
      const message = {
        map_x: trainx,
        map_y: trainy,
        gamex: gamex,
        gamey: gamey,
      };
      const jsonMessage = JSON.stringify(message);
      ws.send(jsonMessage);
    }
  }, [trainx, trainy]);

  useEffect(() => {
    if (!map.current) return; // 确保地图已加载
    // 解析输入的坐标
    const mlng = parseFloat(mapx);
    const mlat = parseFloat(mapy);

    // 如果标记不存在，创建一个新的
    if (!markerRef.current) {
      markerRef.current = new mapboxgl.Marker({
        color: "#FF0000",
        scale: 1,
      })
        .setLngLat([mlng, mlat])
        .addTo(map.current);
    }

    // 检查坐标是否有效
    if (!isNaN(mlng) && !isNaN(mlat)) {
      // 更新标记位置
      markerRef.current.setLngLat([mlng, mlat]);

      // 可选：将地图中心移动到新位置
      map.current.flyTo({
        center: [mlng, mlat],
        zoom: 12, // 可以根据需要调整缩放级别
      });
    }
  }, [mapx, mapy]); // 当 inputLng 或 inputLat 改变时，这个 effect 会重新运行

  useEffect(() => {
    if (map.current) return; // initialize map only once
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      center: [-0.7, 0.7],
      zoom: 10,
      style: {
        version: 8,
        glyphs: "/{fontstack}/{range}.pbf",
        sources: {
          tileMap: {
            type: "raster",
            name: "tileMap",
            tiles: [HEY_TILE_PATH],
            tileSize: 256,
            minZoom: 8, // 最小缩放级别
            maxZoom: 12, // 最大缩放级别
          },
        },
        layers: [
          {
            id: "tileMap",
            type: "raster",
            source: "tileMap",
            layout: {
              visibility: "visible",
            },
            paint: { "background-color": "rgba(0,0,0,0)" },
          },
        ],
      },
      minZoom: 8, // 最小缩放级别
      maxZoom: 12, // 最大缩放级别
      bounds: [-1.4, 0, 0, 1.4],
      maxBounds: [-1.4, 0, 0, 1.4],
    });

    map.current.on("load", () => {
      console.log("Map loaded");

      heibox.map_point.forEach((item) => {
        if (item.map_key !== "chapter1") return;
        const coordinates = proj4("EPSG:3857", "EPSG:4326", item.position);
        map.current.loadImage(`/icons/${item.type}.jpg`, (error, image) => {
          if (error) {
            console.log(`Error loading image:/icons/${item.type}.jpg`, error);
            return;
          }
          map.current.addImage(item.type + item.id, image);
          map.current.addLayer({
            id: item.type + item.id,
            type: "symbol",
            source: {
              type: "geojson",
              data: {
                type: "FeatureCollection",
                features: [
                  {
                    type: "Feature",
                    geometry: {
                      type: "Point",
                      coordinates: coordinates,
                    },
                    properties: {
                      name: item.name,
                    },
                  },
                ],
              },
            },
            layout: {
              "icon-image": item.type + item.id,
              "icon-size": 0.1,
              "text-field": ["get", "name"], // 使用feature的name属性作为文字
              "text-ignore-placement": !0,
              "text-allow-overlap": !0,
              "text-font": ["Open Sans Regular"],
              "text-offset": [0, 1.25], // 相对于图标的偏移量
              "text-anchor": "top", // 文字锚点位置
              "text-size": 12, // 文字大小
            },
            paint: {
              "text-color": "#FFFFFF", // 文字颜色
            },
          });
        });
      });
    });

    map.current.on("click", (e) => {
      var I = e.lngLat.lat,
        E = e.lngLat.lng;
      console.log("当前点击坐标", [E, I]);
    });
    map.current.on("contextmenu", (e) => {
      e.preventDefault(); // Prevent the default context menu
      var I = e.lngLat.lat,
        E = e.lngLat.lng;
      console.log("当前右键点击坐标", [E, I]);
      setTrainx(E);
      setTrainy(I);
    });
  });

  const handleCleanData = async () => {
    if (window.confirm('Are you sure you want to clean all data? This action cannot be undone.')) {
      try {
        const response = await axios.post('http://localhost:8001/clean_data');
        alert(response.data.message);
      } catch (error) {
        console.error('Error cleaning data:', error);
        alert('Failed to clean data. Please try again.');
      }
    }
  };

  return (
    <div style={{ position: 'relative' }}>
      <div ref={mapContainer} className="map-container" />
      <button 
        onClick={handleCleanData} 
        style={{
          position: 'absolute',
          bottom: '20px',
          right: '20px',
          padding: '10px 20px',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
          zIndex: 1000
        }}
      >
        Clean All Data
      </button>
    </div>
  );
}

export default App;
