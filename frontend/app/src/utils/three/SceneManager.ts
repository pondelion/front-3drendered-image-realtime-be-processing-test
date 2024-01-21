import * as THREE from "three";
import { ThreeObject } from "../../types/Three";
import { OrbitControls } from "./OrbitControls";

export class SceneManager {
  public renderer: THREE.WebGLRenderer = new THREE.WebGLRenderer({
    antialias: true,
  });
  private scene: THREE.Scene = new THREE.Scene();
  private objects: ThreeObject[] = [];
  private ambientLight: THREE.AmbientLight | null = new THREE.AmbientLight(
    0xffffff,
    0.5
  );
  private spotLight: THREE.SpotLight | null = new THREE.SpotLight(
    0xffffff,
    1.9,
    100,
    Math.PI / 4,
    1
  );
  private spotLightPos: THREE.Vector3 = new THREE.Vector3(2, 7, 2);
  private camera: THREE.PerspectiveCamera;
  private width: number;
  private height: number;
  private cameraPos: THREE.Vector3 = new THREE.Vector3(5, 5, 5);
  private cameraLookAt: THREE.Vector3 = new THREE.Vector3(0, 0, 0);
  private bgColor: THREE.Color = new THREE.Color("#b4bad2");
  private animationCnt: number = 0;
  private container: HTMLDivElement;
  private orbitControls: OrbitControls | null = null;
  private onFrameCallback: (cnt: number) => void = (cnt: number) => {};
  private paused: boolean = false;

  constructor(
    cameraWidth: number,
    cameraHeight: number,
    container: HTMLDivElement,
    onFrameCallback?: (cnt: number) => void
  ) {
    if (this.spotLight!.castShadow) {
      this.spotLight!.castShadow = true;
    }
    this.width = cameraWidth;
    this.height = cameraHeight;
    this.container = container;
    if (onFrameCallback) {
      this.onFrameCallback = onFrameCallback;
    }
    this.camera = this.resetCamera();
    this.resetAll();
  }

  public resetObjects() {
    // Reset objects
    this.scene.remove.apply(this.scene, this.scene.children);
    // setObjects(objs);
    this.objects.map((obj) => this.scene.add(obj.obj));

    if (this.ambientLight) {
      this.scene.add(this.ambientLight);
    }

    if (this.spotLight) {
      this.spotLight.position.set(
        this.spotLightPos.x,
        this.spotLightPos.y,
        this.spotLightPos.z
      );
      this.scene.add(this.spotLight);
    }
  }

  public resetAll() {
    this.camera = this.resetCamera();

    this.renderer!.setSize(this.width, this.height);
    this.renderer!.setClearColor(this.bgColor);
    this.renderer!.shadowMap.enabled = true;
    this.renderer!.xr.enabled = true;

    if (this.container) {
      if (this.container.childNodes.length === 0) {
        this.container.appendChild(this.renderer!.domElement);
      }
      this.orbitControls = new OrbitControls(this.camera, this.container);
      this.orbitControls.target.set(0, 0, 0);
      this.orbitControls.update();
    }

    this.animationCnt = 0;
    this.renderer!.setAnimationLoop(() => {
      if (this.paused) {
        return;
      }
      this.animationCnt += 1;
      // update(clock.getDelta());
      this.renderer!.render(this.scene, this.camera);
      this.onFrameCallback(this.animationCnt);
    });

    this.resetObjects();
  }

  public setObjects(objs: ThreeObject[], resetScene: boolean = true) {
    this.objects = objs;
    if (resetScene) {
      this.resetObjects();
    }
  }

  public getObjects(): ThreeObject[] {
    return this.objects;
  }

  public resetCamera(): THREE.PerspectiveCamera {
    const camera = new THREE.PerspectiveCamera(
      45,
      this.width / this.height,
      0.1,
      1000
    );
    camera.position.set(this.cameraPos.x, this.cameraPos.y, this.cameraPos.z);
    camera.lookAt(this.cameraLookAt);
    return camera;
  }

  public setPaused(paused: boolean) {
    this.paused = paused;
  }

  public findThreeObj(tag: string): ThreeObject | null {
    if (this.objects === undefined) {
      return null;
    }
    for (let i = 0; i < this.objects?.length; ++i) {
      if (this.objects[i].tag === tag) {
        return this.objects[i];
      }
    }
    return null;
  }

  public getAnimationCount(): number {
    return this.animationCnt;
  }
}
