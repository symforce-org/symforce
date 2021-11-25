import LcmMsg, { LcmType } from "./lcm";

export type PrimitiveInitializer = NumberConstructor | BooleanConstructor | StringConstructor;

export type StructInitializer = LcmType<LcmMsg>;

// Arrays in LCM can be automatically sized, fixed size, or dynamic.
// Auto-sized arrays are only allowed for 1-dimensional arrays.
// Fixed and dynamic array dimensions can be mixed in a n-dimensional array
export interface ArrayDimAuto {
  kind: "auto";
}

export interface ArrayDimFixed {
  kind: "fixed";
  size: number; // the exact size of the array dimension
}

export interface ArrayDimDynamic {
  kind: "dynamic";
  member: string; // name of the member that defines the size
}

export type ArrayDim = ArrayDimAuto | ArrayDimFixed | ArrayDimDynamic;

// In LCM, struct members can be primitives, structs, or arrays of the former two.
export interface ReflectionTypePrimitive {
  kind: "primitive";
  type: PrimitiveInitializer;
  typeStr: string;
}

// Enums are structs, but subclass from LcmEnumMsg, which allows for runtime inspection
// Which "struct" members are enums is not something we know at LCM compile time.
export interface ReflectionTypeStruct {
  kind: "struct";
  type: LcmType<LcmMsg>;
  typeStr: string;
}

export interface ReflectionTypeArray {
  kind: "array";
  dims: ArrayDim[];
  nDim: number; // number of dimensions
  // constructor for type contained within array
  inner: ReflectionTypePrimitive | ReflectionTypeStruct;
}

// NOTE(danny): this is a special case to optimize for byte arrays by using Uint8Array, only for JS
export interface ReflectionTypeBytes {
  kind: "bytes";
  dims: ArrayDim[];
  nDim: 1;
  // constructor is always Uint8Array
}

// All possible types for a struct member
export type ReflectionType =
  | ReflectionTypePrimitive
  | ReflectionTypeStruct
  | ReflectionTypeArray
  | ReflectionTypeBytes;

// Describes the type of the members of an LCM struct message
export interface ReflectionMetaMap {
  [member: string]: ReflectionType;
}

// NOTE(danny): technically, an enum can be sent as the value of a channel, which is an LcmMsg, so
// LcmMsg needs to be able to represent that in the _reflection_meta attribute of LcmMsg
type ReflectionMeta = ReflectionMetaMap | "enum";
export default ReflectionMeta;
