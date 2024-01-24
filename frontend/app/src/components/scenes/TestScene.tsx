import React, { useEffect } from "react";
import Grid from "@mui/material/Grid";
import * as THREE from "three";
import { ThreeObject } from "../../types/Three";
import { ObjectFactory as OF } from "../../utils/three/ObjectFactory";
import { SceneManager } from "../../utils/three/SceneManager";

function computeScreenSpaceBoundingBox(mesh: any, camera: any) {
  if (mesh.geometry.isBufferGeometry) {
    var vertices = mesh.geometry.attributes.position;
  } else {
    var vertices = mesh.geometry.vertices;
  }
  var vertex = new THREE.Vector3();
  var min = new THREE.Vector3(1, 1, 1);
  var max = new THREE.Vector3(-1, -1, -1);
  for (var i = 0; i < vertices.count; i++) {
    vertex.fromBufferAttribute(vertices, i);
    var vertexWorldCoord = vertex.applyMatrix4(mesh.matrixWorld);
    var vertexScreenSpace = vertexWorldCoord.project(camera);
    min.min(vertexScreenSpace);
    max.max(vertexScreenSpace);
  }

  // values are normalized to -1 ~ 1
  // return new THREE.Box2(
  //   new THREE.Vector2(min.x, min.y),
  //   new THREE.Vector2(max.x, max.y)
  // );

  return {
    top_left_x: 0.5 * (min.x + 1.0),
    top_left_y: 0.5 * (min.y + 1.0),
    bottom_right_x: 0.5 * (max.x + 1.0),
    bottom_right_y: 0.5 * (max.y + 1.0),
  };
}

// function normalizedToPixels(coord, renderWidthPixels, renderHeightPixels) {
//   var halfScreen = new THREE.Vector2(renderWidthPixels/2, renderHeightPixels/2)
//   return coord.clone().multiply(halfScreen);
// }

let sceneManager: SceneManager;
let dupSceneManager: SceneManager;
let ws: WebSocket | null = null;
let receivedImageUrl: string | null = null;
let receivedSegImageUrl: string | null = null;
let prevTimestamp = Date.now();

export interface Props {}

const TestScene: React.FC<Props> = (props: Props) => {
  const width: number = 700;
  const height: number = 500;

  const [container, setContainer] = React.useState<HTMLDivElement | null>(null);
  const [dupContainer, setDupContainer] = React.useState<HTMLDivElement | null>(
    null
  );
  const [clock, setClock] = React.useState<THREE.Clock>(new THREE.Clock());
  const [nParticles, setNParticles] = React.useState<number>(25);
  const [imageUrl, setImageUrl] = React.useState<string | undefined>();
  const [imageSegUrl, setImageSegUrl] = React.useState<string | undefined>();
  // const [receivedImageUrl, setReceivedImageUrl] = React.useState<
  //   string | undefined
  // >();
  const [wsCnt, setWsCnt] = React.useState<number>(0);
  const [wsConnected, setWsConnected] = React.useState<boolean>(false);
  const [bboxes, setBboxes] = React.useState<any>([]);

  const createObjects = (nParticles: number): ThreeObject[] => {
    const XYZ_MAX = 7.0;
    let objs: ThreeObject[] = [...new Array(nParticles)].map((v, idx) => {
      return {
        tag: `target${idx}`,
        obj: OF.createSphere(
          2.0 * (XYZ_MAX - 1.0) * (Math.random() - 0.5),
          2.0 * (XYZ_MAX - 1.0) * (Math.random() - 0.5),
          2.0 * (XYZ_MAX - 1.0) * (Math.random() - 0.5),
          0.5,
          0.8,
          "#FF0077",
          THREE.FrontSide
        ),
        objType: "sphere",
      };
    });
    objs = objs.concat(
      [...new Array(nParticles)].map((v, idx) => {
        return {
          tag: `non_target${idx}`,
          obj: OF.createBox(
            2.0 * (XYZ_MAX - 1.0) * (Math.random() - 0.5),
            2.0 * (XYZ_MAX - 1.0) * (Math.random() - 0.5),
            2.0 * (XYZ_MAX - 1.0) * (Math.random() - 0.5),
            0.8,
            0.6,
            0.5,
            0.8,
            "#55FF11",
            THREE.FrontSide
          ),
          objType: "box",
        };
      })
    );
    objs.push({
      tag: "bg",
      obj: OF.createBufferGeometryPlane(10, 50, 1.0, 0xffffff, 0.5),
      objType: "buffer_geometry",
    });
    return objs;
  };

  const onFrameCallback = (cnt: number) => {
    const imgUrl: string = sceneManager.renderer.domElement.toDataURL();
    setImageUrl(imgUrl);

    const bboxes = sceneManager.getObjects().map((obj, index) => {
      const bbox2d = computeScreenSpaceBoundingBox(
        obj.obj,
        sceneManager.getCamera()
      );
      // console.log(bbox2d);
      return { bbox: bbox2d, obj_tag: obj.tag };
    });
    setBboxes(bboxes);
    // console.log(bboxes);
  };

  const onFrameCallbackDup = (cnt: number) => {
    const imgSegUrl: string = dupSceneManager.renderer.domElement.toDataURL();
    // console.log(imgSegUrl);
    setImageSegUrl(imgSegUrl);
  };

  const connectWs = () => {
    if (ws === null) {
      console.log(`connecting : ${wsCnt}`);
      console.log(ws);
      const socket = new WebSocket("ws://172.26.229.147:8000/image_process");
      // setWs(socket);
      ws = socket;
      socket.onmessage = (msg: any) => {
        console.log("recv");
        const data: any = JSON.parse(msg.data);
        if ("image" in data) {
          // setReceivedImageUrl(data.image);
          receivedImageUrl = data.image;
        }
        if ("seg_image" in data) {
          receivedSegImageUrl = data.seg_image;
        }
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
    dupSceneManager = new SceneManager(
      width,
      height,
      dupContainer!,
      onFrameCallbackDup
    );
    dupSceneManager.setCamera(sceneManager.getCamera());
    dupSceneManager.setObjects(createObjects(nParticles));
    dupSceneManager.getObjects().map((dupObj, index) => {
      const obj = sceneManager.getObjects()[index];
      dupObj.obj.position.x = obj.obj.position.x;
      dupObj.obj.position.y = obj.obj.position.y;
      dupObj.obj.position.z = obj.obj.position.z;
      // if (dupObj.obj instanceof THREE.Mesh && obj instanceof THREE.Mesh) {
      //   (dupObj.obj.material as THREE.MeshLambertMaterial).color = (
      //     obj.material as THREE.MeshLambertMaterial
      //   ).color;
      // }
      if (dupObj.obj instanceof THREE.Mesh && obj.obj instanceof THREE.Mesh) {
        // replace with monotonic color material for segmentation annotation
        if (obj.tag?.startsWith("target")) {
          dupObj.obj.material = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
          });
        } else {
          dupObj.obj.material = new THREE.MeshBasicMaterial({
            color: 0x0000ff,
          });
        }
      }
    });
  }, [container, clock]);

  useEffect(() => {
    const now = Date.now();
    const dt_secs = (now - prevTimestamp) / 1000;
    if (dt_secs < 0.2) {
      return;
    }
    if (ws !== null && ws.readyState === ws.OPEN) {
      console.log("sent");
      ws.send(
        JSON.stringify({
          image: imageUrl,
          seg_image: imageSegUrl,
          bboxes: bboxes,
        })
      );
      if (receivedImageUrl !== null) {
        const img = new Image();
        const rcvImageCanvas: any = document.getElementById("rcv_image_canvas");
        const ctx = rcvImageCanvas.getContext("2d");
        img.onload = function () {
          ctx.drawImage(img, 0, 0); // Or at whatever offset you like
        };
        img.src = receivedImageUrl;
      }
      if (receivedSegImageUrl !== null) {
        const img = new Image();
        const rcvImageCanvas2: any =
          document.getElementById("rcv_image_canvas2");
        const ctx = rcvImageCanvas2.getContext("2d");
        img.onload = function () {
          ctx.drawImage(img, 0, 0); // Or at whatever offset you like
        };
        img.src = receivedSegImageUrl;
      }
      prevTimestamp = now;
    }
  }, [imageUrl, imageSegUrl]);

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
        <Grid item xs={6}>
          <div
            style={{ width: width, height: height }}
            ref={(container) => {
              setDupContainer(container);
            }}
          />
        </Grid>
        <Grid item xs={6}>
          <canvas
            width={width}
            height={height}
            id="rcv_image_canvas2"
            // style={{ display: "none" }}
          ></canvas>
        </Grid>
      </Grid>
    </div>
  );
};

export default TestScene;
