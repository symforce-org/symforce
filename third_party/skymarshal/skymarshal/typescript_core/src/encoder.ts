import Long from "long";
import {
  byte,
  int8_t,
  int16_t,
  int32_t,
  int64_t,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  float,
  double
} from "./types";

const USE_LE = false;

type NDimensionalArray<T> = T | ArrayValue<T>;
interface ArrayValue<T> extends Array<NDimensionalArray<T>> {}

export default class Encoder {
  public offset: number;
  public view: DataView;

  constructor(readonly max_size: number) {
    this.view = new DataView(new ArrayBuffer(max_size));
    this.offset = 0;
  }

  public encode_hash(value: Long): void {
    this.encode_long(value);
  }

  public encode_long(value: Long): void {
    this.view.setInt32(this.offset, value.getHighBits(), USE_LE);
    this.offset += 4;
    this.view.setInt32(this.offset, value.getLowBits(), USE_LE);
    this.offset += 4;
  }

  public encode_int64_t(value: int64_t): void {
    this.encode_long(Long.fromNumber(value));
  }

  public encode_int32_t(value: int32_t): void {
    this.view.setInt32(this.offset, value, USE_LE);
    this.offset += 4;
  }

  public encode_int16_t(value: int16_t): void {
    this.view.setInt16(this.offset, value, USE_LE);
    this.offset += 2;
  }

  public encode_int8_t(value: int8_t): void {
    this.view.setInt8(this.offset, value);
    this.offset += 1;
  }

  public encode_uint64_t(value: uint64_t): void {
    this.encode_long(Long.fromNumber(value, true));
  }

  public encode_uint32_t(value: uint32_t): void {
    this.view.setUint32(this.offset, value, USE_LE);
    this.offset += 4;
  }

  public encode_uint16_t(value: uint16_t): void {
    this.view.setUint16(this.offset, value, USE_LE);
    this.offset += 2;
  }

  public encode_uint8_t(value: uint8_t): void {
    this.view.setUint8(this.offset, value);
    this.offset += 1;
  }

  public encode_byte(value: byte): void {
    this.view.setUint8(this.offset, value);
    this.offset += 1;
  }

  public encode_double(value: double): void {
    this.view.setFloat64(this.offset, value, USE_LE);
    this.offset += 8;
  }

  public encode_float(value: float): void {
    this.view.setFloat32(this.offset, value, USE_LE);
    this.offset += 4;
  }

  public encode_boolean(value: boolean): void {
    this.view.setInt8(this.offset, value ? 1 : 0);
    this.offset += 1;
  }

  public encode_string(value: string): void {
    // strings must be encoded as utf-8, js strings are a wacky UTF-16-like encoding
    const encoded = new TextEncoder().encode(value);
    // write size plus null byte
    this.view.setUint32(this.offset, encoded.length + 1, USE_LE);
    this.offset += 4;
    this.encode_uint8_array(encoded);
    // encode null byte
    this.encode_byte(0);
  }

  public encode_uint8_array(value: Uint8Array): void {
    const buffer = this.view.buffer;
    const combinedOffset = this.offset + this.view.byteOffset;

    // length checking
    if (this.view.byteLength < combinedOffset + value.length) {
      throw new Error(
        `not enough data in buffer to encode length ${value.length} array`
      );
    }
    this.offset += value.length;
    const typedArray = new Uint8Array(buffer, combinedOffset, value.length);
    typedArray.set(value);
  }
}
