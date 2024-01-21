import SpriteText from "three-spritetext";
import * as THREE from "three";

// export type Vec2 = {
//   x: number,
//   y: number,
// };
export type Vec2 = THREE.Vector2;

// export type Vec3 = {
//   x: number,
//   y: number,
//   z: number,
// };
export type Vec3 = THREE.Vector3;

export type ObjectType =
  | "box"
  | "sphere"
  | "plane"
  | "sprite"
  | "arrow"
  | "line"
  | "text"
  | "buffer_geometry"
  | "other";

export type ThreeObject = {
  tag?: string;
  obj: THREE.Mesh | THREE.ArrowHelper | THREE.Sprite | THREE.Line | SpriteText;
  objType: ObjectType;
};

export type ThreeObjects = ThreeObject[];
