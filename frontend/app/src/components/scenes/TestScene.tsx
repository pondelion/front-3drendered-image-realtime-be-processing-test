import React, { useEffect } from "react";
import Grid from "@mui/material/Grid";
import * as THREE from "three";
import { ThreeObject } from "../../types/Three";
import { ObjectFactory as OF } from "../../utils/three/ObjectFactory";
import { SceneManager } from "../../utils/three/SceneManager";

let sceneManager: SceneManager;
let ws: WebSocket | null = null;
let receivedImageUrl: string | null = null;

export interface Props {}

const TestScene: React.FC<Props> = (props: Props) => {
  const width: number = 800;
  const height: number = 600;

  const [container, setContainer] = React.useState<HTMLDivElement | null>(null);
  const [clock, setClock] = React.useState<THREE.Clock>(new THREE.Clock());
  const [nParticles, setNParticles] = React.useState<number>(500);
  const [imageUrl, setImageUrl] = React.useState<string | undefined>();
  // const [receivedImageUrl, setReceivedImageUrl] = React.useState<
  //   string | undefined
  // >();
  const [wsCnt, setWsCnt] = React.useState<number>(0);
  const [wsConnected, setWsConnected] = React.useState<boolean>(false);

  const createObjects = (nParticles: number): ThreeObject[] => {
    const XYZ_MAX = 7.0;
    const objs: ThreeObject[] = [...new Array(nParticles)].map((v, idx) => {
      return {
        tag: `test${idx}`,
        obj: OF.createSphere(
          2.0 * (XYZ_MAX - 1.0) * (Math.random() - 0.5),
          2.0 * (XYZ_MAX - 1.0) * (Math.random() - 0.5),
          2.0 * (XYZ_MAX - 1.0) * (Math.random() - 0.5),
          0.1,
          0.8,
          Math.random() < 0.5 ? "#FF0000" : "#00FF00",
          THREE.FrontSide
        ),
        objType: "sphere",
      };
    });
    // objs.push({
    //   tag: "bg",
    //   obj: OF.createBufferGeometryPlane(10, 50),
    //   objType: "buffer_geometry"
    // })
    return objs;
  };

  const onFrameCallback = (cnt: number) => {
    const imgUrl: string = sceneManager.renderer.domElement.toDataURL();
    const img = new Image();
    img.src = imgUrl;
    // console.log(img);
    setImageUrl(imgUrl);
    if (ws !== null && ws.readyState === ws.OPEN) {
      console.log("sent");
      ws.send(JSON.stringify({ image: imgUrl }));
      if (receivedImageUrl !== null) {
        const img = new Image();
        const rcvImageCanvas: any = document.getElementById("rcv_image_canvas");
        const ctx = rcvImageCanvas.getContext("2d");
        img.onload = function () {
          ctx.drawImage(img, 0, 0); // Or at whatever offset you like
        };
        img.src = receivedImageUrl;
      }
    }
  };

  const connectWs = () => {
    if (ws === null) {
      console.log(`connecting : ${wsCnt}`);
      console.log(ws);
      const socket = new WebSocket("ws://172.26.229.147:8000/image_process");
      // setWs(socket);
      ws = socket;
      socket.onmessage = (msg: any) => {
        // console.log(msg.data["positions"]);
        const data: any = JSON.parse(msg.data);
        if ("image" in data) {
          // setReceivedImageUrl(data.image);
          receivedImageUrl = data.image;
          // console.log(data.image);
        }
        // console.log(receivedImageUrl);
        // socket.send("ok");
      };
      // setWsCnt(wsCnt+1);
    }
  };

  const closeWs = () => {
    ws?.send("close");
    ws?.close();
    // setWs(null);
    ws = null;
    setWsConnected(false);
  };

  useEffect(() => {
    console.log("useeffect");
    //resetAll();
    sceneManager = new SceneManager(width, height, container!, onFrameCallback);
    sceneManager.setObjects(createObjects(nParticles));
  }, [container, clock]);

  useEffect(() => {
    console.log("useeffect3");
    const objects = sceneManager.getObjects();
  }, []);

  return (
    <div>
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <div
            style={{ width: width, height: height }}
            ref={(container) => {
              setContainer(container);
            }}
          />
          <button onClick={closeWs}>close ws connection</button>
          <button
            onClick={() => {
              connectWs();
            }}
          >
            connect ws
          </button>
          {/* {imageUrl} */}
          {/* <img src={receivedImageUrl}></img> */}
          {receivedImageUrl}
        </Grid>
        <Grid item xs={6}>
          <canvas
            width={width}
            height={height}
            id="rcv_image_canvas"
            // style={{ display: "none" }}
          ></canvas>
        </Grid>
      </Grid>
    </div>
  );
};

export default TestScene;
