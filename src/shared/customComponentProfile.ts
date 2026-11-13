// 独自の部品（customHtml を持つ部品）を、操作シミュレーターがほかの部品と同じ基準で判定できる形に置き換える。
// 部品が何のためのものかの読み取りだけを Gemini に任せ（サーバーの customComponentProfiles.ts）、
// 読み取った結果からシミュレーター用の部品を組み立てるのは、このファイルのコードの規則で行う。
// 同じ読み取り結果なら、必ず同じ部品と同じ座標になる

import { buildActionsFromDataAttrs, isSafeStateKey } from './appRuntime.ts';
import type {
  AppComponent,
  AppSpec,
  ComponentAction,
  ComponentBoundsInput,
  ComponentSize,
} from './types.ts';

/** 独自の部品が、どの部品の代わりになるか */
export const CUSTOM_COMPONENT_PURPOSES = [
  'order',
  'coupon',
  'subscription',
  'input_shortcut',
  'required_input',
  'total',
  'info',
  'ad',
  'danger',
  'decoy',
  'other',
] as const;

export type CustomComponentPurpose = (typeof CUSTOM_COMPONENT_PURPOSES)[number];

export const COMPONENT_SIZES: readonly ComponentSize[] = ['xs', 'sm', 'md', 'lg', 'xl'];

/** Gemini が独自の部品を読み取った結果。スキーマで形を決め、シミュレーターが使う値だけを持つ */
export interface CustomComponentProfile {
  purpose: CustomComponentPurpose;
  /** 利用者が自分で押したり選んだりして初めて効くか */
  userChooses: boolean;
  /** 何もしなくても、最初から申し込み・適用された状態になっているか */
  enabledByDefault: boolean;
  /** 判断に必要な文字のうち、一番小さいものの大きさ */
  textSize: ComponentSize;
  /** 押す部分の大きさ。押す部分がなければ表示全体の大きさ */
  tapTargetSize: ComponentSize;
  /** 気づかないと損をする条件（自動で続く契約・解約の制限・追加の料金など）が、小さい字や目立たない場所にあるか */
  hasHiddenCondition: boolean;
  /** 広告などで、閉じる手段が見つけにくいか小さいか */
  hardToClose: boolean;
  /** 偽の残り時間など、急かす表示があるか */
  pressure: boolean;
}

/** 変換の対象になる独自の部品か。customHtml を持つ部品は、部品の種類にかかわらず対象にする */
export function isCustomComponent(comp: AppComponent): boolean {
  return comp.type === 'custom' || Boolean(comp.customHtml);
}

/** 受け取った値が読み取り結果の形をしているか。形が違えば読み取りに失敗したものとして扱う */
export function isCustomComponentProfile(value: unknown): value is CustomComponentProfile {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.purpose === 'string' &&
    (CUSTOM_COMPONENT_PURPOSES as readonly string[]).includes(v.purpose) &&
    typeof v.userChooses === 'boolean' &&
    typeof v.enabledByDefault === 'boolean' &&
    typeof v.textSize === 'string' &&
    (COMPONENT_SIZES as readonly string[]).includes(v.textSize) &&
    typeof v.tapTargetSize === 'string' &&
    (COMPONENT_SIZES as readonly string[]).includes(v.tapTargetSize) &&
    typeof v.hasHiddenCondition === 'boolean' &&
    typeof v.hardToClose === 'boolean' &&
    typeof v.pressure === 'boolean'
  );
}

/** 読み取り結果から、決まった順番のキーだけを持つ値を作る。キャッシュに置く値と比べる値をそろえる */
export function pickCustomComponentProfile(value: CustomComponentProfile): CustomComponentProfile {
  return {
    purpose: value.purpose,
    userChooses: value.userChooses,
    enabledByDefault: value.enabledByDefault,
    textSize: value.textSize,
    tapTargetSize: value.tapTargetSize,
    hasHiddenCondition: value.hasHiddenCondition,
    hardToClose: value.hardToClose,
    pressure: value.pressure,
  };
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      const v = (value as Record<string, unknown>)[key];
      if (v !== undefined) out[key] = stableValue(v);
    }
    return out;
  }
  if (typeof value === 'string') return value.replace(/\s+/g, ' ').trim();
  return value;
}

/**
 * 独自の部品の定義を、キャッシュの鍵に使う決まった文字列にする。
 * キーの順番と空白の違いはそろえ、部品のIDは含めない（同じ中身の部品なら同じ読み取り結果を使う）
 */
export function normalizeCustomComponentForProfile(comp: AppComponent): string {
  const { id: _id, ...rest } = comp;
  return JSON.stringify(stableValue(rest));
}

/** 独自の部品のHTMLに書かれた、押したときの動きを読む */
export function parseCustomHtmlActions(spec: AppSpec, customHtml: string | undefined): ComponentAction[] {
  if (!customHtml) return [];
  const acts: ComponentAction[] = [];
  const attrRe = /data-(set-state|toggle-state|increment-state|complete-task)\s*=\s*(?:"([^"]*)"|'([^']*)')/gi;
  for (const m of customHtml.matchAll(attrRe)) {
    const kind = m[1].toLowerCase();
    const raw = (m[2] ?? m[3] ?? '').trim();
    acts.push(
      ...buildActionsFromDataAttrs(spec, {
        setState: kind === 'set-state' ? raw : undefined,
        toggleState: kind === 'toggle-state' ? raw : undefined,
        incrementState: kind === 'increment-state' ? raw : undefined,
        completeTask: kind === 'complete-task' ? raw : undefined,
      }),
    );
  }
  return acts;
}

/** 動きの中から、状態のキーを1つ選ぶ（定期便の欄なら subscription） */
function stateKeyFromActions(actions: ComponentAction[], prefer?: (key: string) => boolean): string | undefined {
  const keys = actions
    .filter((a) => a.type === 'set_state' || a.type === 'toggle_state')
    .map((a) => a.targetKey)
    .filter((k): k is string => Boolean(k));
  return (prefer ? keys.find(prefer) : undefined) ?? keys[0];
}

/** 独自の部品のHTMLで、入力欄やチェック欄が状態と結び付いているキー（data-state-key） */
function stateKeysFromCustomHtml(customHtml: string | undefined): string[] {
  if (!customHtml) return [];
  return [...customHtml.matchAll(/data-state-key\s*=\s*(?:"([^"]*)"|'([^']*)')/gi)]
    .map((m) => (m[1] ?? m[2] ?? '').trim())
    .filter((key) => Boolean(key) && isSafeStateKey(key));
}

/** いつでも当てはまる条件。状態のキーが分からない隠れた条件を、罠として扱うために使う */
const ALWAYS_TRUE_CONDITION = { stateKey: '__customHiddenCondition', operator: 'neq', value: '__never' } as const;

/**
 * 読み取り結果から、シミュレーター用の部品を組み立てる。
 * 部品のID・表示条件・固定表示などの置き場所はそのまま残し、種類・役割・文字の大きさ・動きを、既存の部品と同じ基準で表す
 */
function buildCustomComponentProxy(
  spec: AppSpec,
  comp: AppComponent,
  profile: CustomComponentProfile,
): AppComponent {
  const actions = comp.actions.length > 0 ? comp.actions : parseCustomHtmlActions(spec, comp.customHtml);
  const style = { ...comp.style, size: profile.textSize };
  const base: AppComponent = {
    ...comp,
    style,
    actions,
    customHtml: undefined,
    customCss: undefined,
    darkPattern: profile.pressure ? 'false_urgency' : comp.darkPattern,
  };
  delete base.customHtml;
  delete base.customCss;
  delete base.trapWhen;
  delete base.trapSafeValue;
  if (!base.darkPattern) delete base.darkPattern;

  const withLabelWord = (word: string, test: RegExp) =>
    test.test(comp.label) ? comp.label : `${word} ${comp.label}`.trim();

  // 広告は、ポップアップのままなら閉じやすさだけを、画面内に置いたものなら関係ない表示として扱う
  if (comp.type === 'popup') {
    return {
      ...base,
      type: 'popup',
      role: 'popup',
      closePosition: profile.hardToClose ? 'top-right' : (comp.closePosition ?? 'bottom-bar'),
      closeSize: profile.hardToClose ? 'xs' : (comp.closeSize ?? 'md'),
      closeLabel: profile.hardToClose ? '×' : (comp.closeLabel ?? '閉じる'),
      label: profile.purpose === 'coupon' ? withLabelWord('クーポン', /クーポン|OFF|割引/) : comp.label,
    };
  }

  // 気づかないと損をする条件は、罠として扱う。細かい字を読む人だけが気づき、読まない人は気づかずに後悔する
  const hiddenTrap = (stateKey: string | undefined): Partial<AppComponent> => ({
    role: 'trap',
    trapWhen: stateKey ? { stateKey, operator: 'eq', value: true } : { ...ALWAYS_TRUE_CONDITION },
    ...(stateKey ? { trapSafeValue: false } : {}),
  });

  switch (profile.purpose) {
    case 'order':
      return { ...base, type: 'button', role: 'primary_action' };
    case 'decoy':
      return { ...base, type: 'button', role: 'decoy' };
    case 'danger':
      return { ...base, type: 'button', role: 'danger' };
    case 'input_shortcut':
      return { ...base, type: 'button', role: 'info' };
    case 'coupon':
      return {
        ...base,
        type: 'button',
        role: profile.hasHiddenCondition ? 'trap' : 'info',
        label: withLabelWord('クーポン', /クーポン|OFF|割引/),
        ...(profile.hasHiddenCondition ? hiddenTrap(undefined) : {}),
      };
    case 'subscription': {
      const isSubKey = (k: string) => k.toLowerCase().includes('subscri');
      const htmlKeys = stateKeysFromCustomHtml(comp.customHtml);
      const stateKey =
        comp.stateKey ??
        stateKeyFromActions(actions, isSubKey) ??
        htmlKeys.find(isSubKey) ??
        htmlKeys[0] ??
        'subscription';
      const initial = spec.state.find((s) => s.key === stateKey)?.initialValue;
      const forced = profile.enabledByDefault || !profile.userChooses || initial === true;
      const label = withLabelWord('定期便', /定期/);
      // 自分で選べず最初から申し込まれている、または条件が隠れているものは罠。正直なものは選べる欄として扱う
      if (forced || profile.hasHiddenCondition) {
        return { ...base, type: 'checkbox', stateKey, label, ...hiddenTrap(stateKey) };
      }
      return { ...base, type: 'checkbox', role: 'info', stateKey, label };
    }
    case 'required_input': {
      const stateKey = comp.stateKey ?? stateKeysFromCustomHtml(comp.customHtml)[0];
      return stateKey
        ? { ...base, type: 'input', role: 'required_input', stateKey }
        : { ...base, type: 'text', role: 'info' };
    }
    case 'total':
      return { ...base, type: 'text', role: 'total' };
    case 'ad':
      return { ...base, type: 'status_banner', role: 'distraction' };
    case 'info':
    case 'other':
    default:
      return {
        ...base,
        type: 'text',
        role: profile.hasHiddenCondition ? 'trap' : 'info',
        ...(profile.hasHiddenCondition ? hiddenTrap(undefined) : {}),
      };
  }
}

export const TAP_HEIGHT_BY_SIZE: Record<ComponentSize, number> = { xs: 26, sm: 34, md: 44, lg: 56, xl: 68 };

/**
 * 押す部分が部品全体より小さい独自の部品は、測った枠を押す部分の高さまで縮める（広げることはしない）。
 * 大きな枠の中に小さなボタンがある部品で、枠全体を押せるものとして扱わないようにする
 */
function shrinkToTapTarget(bound: ComponentBoundsInput, size: ComponentSize): ComponentBoundsInput {
  const h = bound.bottom - bound.top;
  const target = TAP_HEIGHT_BY_SIZE[size];
  if (h <= target) return bound;
  const centerY = (bound.top + bound.bottom) / 2;
  return { ...bound, top: Math.round(centerY - target / 2), bottom: Math.round(centerY + target / 2) };
}

export interface SimulationInput {
  spec: AppSpec;
  bounds?: ComponentBoundsInput[];
}

/**
 * 判定に使う画面と座標を作る。読み取り結果がある独自の部品だけをシミュレーター用の部品に置き換え、
 * 読み取り結果がない独自の部品は、そのまま（部品のHTMLに書かれた動きと文字で）扱う
 */
export function prepareSimulationInput(
  spec: AppSpec,
  profiles: Readonly<Record<string, CustomComponentProfile>> | undefined,
  bounds?: ComponentBoundsInput[],
): SimulationInput {
  if (!profiles || Object.keys(profiles).length === 0) return { spec, bounds };
  const lookupProfile = (id: string): CustomComponentProfile | undefined =>
    Object.hasOwn(profiles, id) ? profiles[id] : undefined;
  const components = spec.components.map((comp) => {
    const profile = isCustomComponent(comp) ? lookupProfile(comp.id) : undefined;
    return profile ? buildCustomComponentProxy(spec, comp, profile) : comp;
  });
  const adjustedBounds = bounds?.map((b) => {
    const comp = spec.components.find((c) => c.id === b.id);
    const profile = comp && isCustomComponent(comp) ? lookupProfile(comp.id) : undefined;
    return profile ? shrinkToTapTarget(b, profile.tapTargetSize) : b;
  });
  return { spec: { ...spec, components }, bounds: adjustedBounds };
}
