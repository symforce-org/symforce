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

export default class Decoder {
  public offset: number;
  public view: DataView;

  constructor(encoded: DataView) {
    this.view = encoded;
    this.offset = 0;
  }

  public decode_hash(): Long {
    return this.decode_ulong();
  }

  public decode_long(): Long {
    const value = new Long(
      this.view.getInt32(this.offset + 4, USE_LE),
      this.view.getInt32(this.offset, USE_LE)
    );
    this.offset += 8;
    return value;
  }

  public decode_ulong(): Long {
    const value = new Long(
      this.view.getInt32(this.offset + 4, USE_LE),
      this.view.getInt32(this.offset, USE_LE),
      true
    );
    this.offset += 8;
    return value;
  }

  public decode_int64_t(): int64_t {
    return this.decode_long().toNumber();
  }

  public decode_int32_t(): int32_t {
    const value = this.view.getInt32(this.offset, USE_LE);
    this.offset += 4;
    return value;
  }

  public decode_int16_t(): int16_t {
    const value = this.view.getInt16(this.offset, USE_LE);
    this.offset += 2;
    return value;
  }

  public decode_int8_t(): int8_t {
    const value = this.view.getInt8(this.offset);
    this.offset += 1;
    return value;
  }

  public decode_uint64_t(): uint64_t {
    return this.decode_long().toNumber();
  }

  public decode_uint32_t(): uint32_t {
    const value = this.view.getUint32(this.offset, USE_LE);
    this.offset += 4;
    return value;
  }

  public decode_uint16_t(): uint16_t {
    const value = this.view.getUint16(this.offset, USE_LE);
    this.offset += 2;
    return value;
  }

  public decode_uint8_t(): uint8_t {
    const value = this.view.getUint8(this.offset);
    this.offset += 1;
    return value;
  }

  public decode_byte(): byte {
    const value = this.view.getUint8(this.offset);
    this.offset += 1;
    return value;
  }

  public decode_double(): double {
    const value = this.view.getFloat64(this.offset, USE_LE);
    this.offset += 8;
    return value;
  }

  public decode_float(): float {
    const value = this.view.getFloat32(this.offset, USE_LE);
    this.offset += 4;
    return value;
  }

  public decode_boolean(): boolean {
    const value = this.view.getInt8(this.offset);
    this.offset += 1;
    return Boolean(value);
  }

  public decode_string(): string {
    const stringLen = this.view.getUint32(this.offset, USE_LE);
    this.offset += 4;
    // don't decode null byte
    const value = this.decode_uint8_array([stringLen - 1]);
    this.offset += 1; // offset for null byte
    // TODO(danny): needs polyfill for Edge/IE
    return new TextDecoder("utf-8").decode(value);
  }

  public decode_uint8_array(dimensions: number[]): Uint8Array {
    const size = dimensions.reduce((accumulated, value) => {
      return accumulated * value;
    });

    const combinedOffset = this.offset + this.view.byteOffset;

    // length checking
    if (this.view.byteLength + this.view.byteOffset < combinedOffset + size) {
      throw new Error(
        `not enough data in buffer to decode length ${size} array`
      );
    }

    this.offset += size;
    return new Uint8Array(this.view.buffer, combinedOffset, size);
  }
}
