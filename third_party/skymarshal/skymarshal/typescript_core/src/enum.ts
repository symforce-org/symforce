import LcmMsg from "./lcm";
import ReflectionMeta from "./reflection";

type TypeScriptEnum = Record<string, string | number> & { [k: number]: string };

export default class LcmEnumMsg extends LcmMsg {
  // https://github.com/microsoft/TypeScript/issues/3841#issuecomment-502845949
  ["constructor"]!: typeof LcmEnumMsg; // allow access to constructor

  public static option_t: TypeScriptEnum;
  public static _reflection_meta: ReflectionMeta = "enum";

  constructor(value?: number) {
    super();
    this.value = value || 0;
  }

  public value: number;
  public get name(): string {
    throw Error("Base enum doesn't have a name");
  }
}
