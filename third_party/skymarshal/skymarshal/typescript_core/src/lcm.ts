import Decoder from "./decoder";
import Encoder from "./encoder";
import ReflectionMeta from "./reflection";
import Long from "long";

export type LcmType<D extends LcmMsg> = new () => D;

export default class LcmMsg {
  ["constructor"]!: typeof LcmMsg; // allow for access to constructor
  public static _get_packed_fingerprint(): Long {
    throw Error("Can't compute packed fingerprint of base class!");
  }

  public static _get_hash_recursive(parents: LcmMsg[]): Long {
    throw Error("Can't compute hash of base class!");
  }

  // type information for runtime reflection
  public static _reflection_meta: ReflectionMeta;

  // initialize all fields to undefined
  public decode(buf: ArrayBuffer): LcmMsg {
    throw Error("Can't decode base class!");
  }

  public decode_one(decoder: Decoder): LcmMsg {
    throw Error("Can't decode base class!");
  }

  public encode(): ArrayBuffer {
    throw Error("Can't encode base class!");
  }

  public encode_one(encoder: Encoder): void {
    throw Error("Can't encode base class!");
  }
}
