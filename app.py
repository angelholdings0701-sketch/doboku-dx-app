import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import unicodedata
import re
import os
from streamlit_gsheets import GSheetsConnection

# === ページ設定 ===
st.set_page_config(page_title="工事実績管理DB", layout="wide")

# === data_editor カスタマイズ（Enterキー + メニュー日本語化 + 項目追加/削除連携） ===
components.html("""
<script>
(function() {
    const doc = window.parent.document;

    // --- 2回Enterで確定 ---
    let lastEnterTime = 0;
    doc.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter' || e.shiftKey || e.ctrlKey || e.metaKey) return;
        const active = doc.activeElement;
        if (!active || !active.classList.contains('gdg-input')) return;
        const now = Date.now();
        if (now - lastEnterTime > 500) {
            e.preventDefault();
            e.stopImmediatePropagation();
            lastEnterTime = now;
        } else {
            lastEnterTime = 0;
        }
    }, true);

    // --- アクティブタブ内のStreamlitボタンをクリック ---
    function clickButtonInActiveTab(searchText) {
        const panels = doc.querySelectorAll('[role="tabpanel"]');
        for (const panel of panels) {
            if (panel.hidden) continue;
            const buttons = panel.querySelectorAll('button');
            for (const btn of buttons) {
                if (btn.textContent.includes(searchText)) {
                    btn.click();
                    return;
                }
            }
        }
    }

    // --- カラムヘッダーメニュー：日本語化 + 項目追加/削除ボタン ---
    const obs = new MutationObserver(function(muts) {
        for (const m of muts) {
            for (const node of m.addedNodes) {
                if (node.nodeType !== 1) continue;
                const txt = node.textContent || '';
                if (txt.includes('Autosize') && txt.includes('Hide column')) {
                    requestAnimationFrame(() => {
                        if (node.dataset.customized) return;
                        node.dataset.customized = '1';

                        // 1. テキスト翻訳
                        const trans = {
                            'Autosize':'列幅を自動調整',
                            'Pin column':'列を固定',
                            'Hide column':'列を非表示',
                            'Format':'書式'
                        };
                        let hideItem = null;
                        const walker = doc.createTreeWalker(node, NodeFilter.SHOW_TEXT);
                        while (walker.nextNode()) {
                            const t = walker.currentNode.textContent.trim();
                            if (trans[t]) {
                                walker.currentNode.textContent = walker.currentNode.textContent.replace(t, trans[t]);
                                if (t === 'Hide column') {
                                    let p = walker.currentNode.parentElement;
                                    while (p && p !== node) {
                                        if (p.parentElement === node || p.parentElement?.children.length > 2) {
                                            hideItem = p; break;
                                        }
                                        p = p.parentElement;
                                    }
                                    if (!hideItem) hideItem = walker.currentNode.parentElement;
                                }
                            }
                        }
                        if (!hideItem) return;

                        // 2. メニューアイテム追加
                        function makeItem(label, color, onClick) {
                            const item = hideItem.cloneNode(true);
                            const w2 = doc.createTreeWalker(item, NodeFilter.SHOW_TEXT);
                            let done = false;
                            while (w2.nextNode()) {
                                if (w2.currentNode.textContent.includes('列を非表示') && !done) {
                                    w2.currentNode.textContent = label;
                                    done = true;
                                }
                            }
                            const svg = item.querySelector('svg');
                            if (svg) svg.remove();
                            item.style.color = color;
                            item.addEventListener('click', function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                onClick();
                            });
                            return item;
                        }

                        // 3. メニューから列名を取得
                        function getColumnName() {
                            // Glide Data Grid メニュー上部の入力フィールドから列名を取得
                            var inp = node.querySelector('input');
                            if (inp && inp.value && inp.value.trim()) {
                                return inp.value.trim();
                            }
                            // フォールバック: span から探す
                            const skip = ['列幅を自動調整','列を固定','列を非表示','書式',
                                          '🗑 項目を削除','➕ 項目を追加',
                                          'tag','tags','notes','text','number','boolean',
                                          'Autosize','Pin column','Hide column','Format'];
                            const spans = node.querySelectorAll('span');
                            for (const s of spans) {
                                const t = s.textContent.trim();
                                if (t && t.length > 0 && t.length < 30 && !skip.includes(t) && s.children.length === 0) {
                                    return t;
                                }
                            }
                            return '';
                        }

                        // 4. ダイアログを開き、セレクトボックスで列を自動選択
                        function openDeleteAndSelect(colName) {
                            clickButtonInActiveTab('項目を削除');
                            if (!colName) return;
                            // ダイアログが開くのを待ってセレクトボックスを操作
                            var attempts = 0;
                            var checker = setInterval(function() {
                                attempts++;
                                if (attempts > 50) { clearInterval(checker); return; }
                                var dialog = doc.querySelector('[role="dialog"]');
                                if (!dialog) return;
                                var sbInput = dialog.querySelector('input[role="combobox"]');
                                if (!sbInput) return;
                                clearInterval(checker);
                                // セレクトボックスをクリックして開く
                                sbInput.focus();
                                sbInput.click();
                                setTimeout(function() {
                                    // 列名を入力してフィルタ
                                    var nativeSetter = Object.getOwnPropertyDescriptor(
                                        window.parent.HTMLInputElement.prototype, 'value'
                                    ).set;
                                    nativeSetter.call(sbInput, colName);
                                    sbInput.dispatchEvent(new Event('input', { bubbles: true }));
                                    // 一致するオプションをクリック
                                    setTimeout(function() {
                                        var opts = doc.querySelectorAll('[role="option"]');
                                        for (var i = 0; i < opts.length; i++) {
                                            if (opts[i].textContent.trim() === colName) {
                                                opts[i].click();
                                                return;
                                            }
                                        }
                                        // 完全一致がなければ部分一致
                                        for (var j = 0; j < opts.length; j++) {
                                            if (opts[j].textContent.includes(colName) || colName.includes(opts[j].textContent.trim())) {
                                                opts[j].click();
                                                return;
                                            }
                                        }
                                    }, 300);
                                }, 200);
                            }, 100);
                        }

                        const delBtn = makeItem('🗑 項目を削除', '#ff6b6b', function() {
                            openDeleteAndSelect(getColumnName());
                        });
                        hideItem.parentElement.insertBefore(delBtn, hideItem.nextSibling);

                        const addBtn = makeItem('➕ 項目を追加', '#51cf66', function() {
                            clickButtonInActiveTab('項目を追加');
                        });
                        hideItem.parentElement.insertBefore(addBtn, delBtn.nextSibling);
                    });
                }
            }
        }
    });
    obs.observe(doc.body || doc.documentElement, { childList: true, subtree: true });
})();
</script>
""", height=0, width=0)

# ==========================================
# 🔒 セキュリティ設定（簡易パスワード認証）
# ==========================================
def check_password():
    """パスワード認証が通っていなければ入力を求め、停止する"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # 認証済みなら何もしない
    if st.session_state.authenticated:
        return True

    # 画面表示
    st.title("🔒 ログインが必要です")

    # secrets.tomlにパスワードがない場合の安全策
    if "PASSWORD" not in st.secrets:
        st.error("管理者に連絡してください（設定ファイルにパスワードが未設定です）")
        st.stop()

    password_input = st.text_input("パスワードを入力してください", type="password")
    if st.button("ログイン"):
        if password_input == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()  # 画面をリロードしてメイン処理へ
        else:
            st.error("パスワードが違います")
    return False

# 認証チェック実行（失敗または未入力ならここでプログラムが止まる）
if not check_password():
    st.stop()

# ==========================================
# 🚀 ここからメインアプリ
# ==========================================

# === Google Sheets の各シート名 ===
KOUJI_SHEET = "dobokudata"  # IDではなく名前にする
ENGINEER_SHEET = "engineer_list"  # IDではなく名前にする

st.title("📋 技術者・工事実績管理データベース")
st.sidebar.success("✅ ログイン中")

# === GSheets 接続 ===
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error(f"❌ 接続エラー: {e}")
    st.stop()

# =========================
# データ処理用関数
# =========================
def normalize_text(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    return text.lower()

def clean_string_for_match(text):
    if pd.isnull(text):
        return ""
    norm = normalize_text(text)
    return norm.replace(" ", "").replace(" ", "")

def process_price_data(x):
    if pd.isnull(x) or str(x).strip() == "":
        return 0
    s_clean = normalize_text(x)
    s_clean = s_clean.replace(",", "").replace("円", "")
    numbers = re.findall(r"\d+", s_clean)
    if not numbers:
        return 0
    valid_nums = []
    for n in numbers:
        if len(n) > 15:
            continue
        valid_nums.append(int(n))
    if not valid_nums:
        return 0
    return max(valid_nums)

# ==========================================
# 数量キーワード検索用の定義と関数
# ==========================================

# 選択可能なキーワード一覧（キーワード名 → (単位, ステップ幅)）
QUANTITY_KEYWORDS = {
    "工事延長": ("m", 100),
    "掘削工": ("m3", 100),
    "掘削": ("m3", 100),
    "盛土": ("m3", 100),
    "盛土工": ("m3", 100),
    "路体盛土工": ("m3", 100),
    "残土処理": ("m3", 100),
    "土砂掘削": ("m3", 100),
    "生コン": ("m3", 10),
    "舗装工": ("m2", 100),
    "表層工": ("m2", 100),
    "基層工": ("m2", 100),
    "上層路盤工": ("m2", 100),
    "下層路盤工": ("m2", 100),
    "路面切削工": ("m2", 100),
    "切削オーバーレイ工": ("m2", 100),
    "歩道舗装工": ("m2", 100),
    "舗装版破砕": ("m2", 100),
    "路床置換工": ("m2", 100),
    "中間層工": ("m2", 100),
    "透水性舗装工": ("m2", 100),
    "薄層カラー舗装工": ("m2", 100),
    "かごマット": ("m2", 100),
    "布設工": ("m", 10),
    "縁石工": ("m", 10),
    "側溝工": ("m", 10),
    "防護柵工": ("m", 10),
    "区画線工": ("m", 10),
    "区画線設置工": ("m", 10),
    "鉄筋": ("t", 1),
    "鉄筋工": ("t", 1),
    "根固ブロック": ("t", 1),
    "型枠工": ("m2", 10),
    "法面整形工": ("m2", 100),
    "鋼矢板": ("枚", 1),
    "集水桝工": ("箇所", 1),
    "マンホール設置工": ("箇所", 1),
}

def extract_quantity_from_text(text, keyword):
    """
    工事概要テキストから、指定キーワードに対応する数値を抽出する。
    複数マッチした場合は最大値を返す。
    """
    if pd.isnull(text):
        return 0.0
    norm = unicodedata.normalize("NFKC", str(text))
    # キーワードの後に続く数値を探す（スペースや記号を許容）
    # 例: "掘削工 1,280m3" → 1280
    # 例: "工事延長：1,760m" → 1760
    # 例: "盛土37400m3" → 37400
    escaped_kw = re.escape(keyword)
    pattern = escaped_kw + r'[^0-9]*?([\d,]+(?:\.\d+)?)'
    matches = re.findall(pattern, norm)
    if not matches:
        return 0.0
    values = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            values.append(val)
        except ValueError:
            continue
    return max(values) if values else 0.0


# =========================
# データの読み込み・保存
# =========================
@st.cache_data(ttl=600)
def load_data():
    # --- 工事データ ---
    core_k_cols = [
        '工事名', '工事概要（主な工種、規格、数量）', '工種名', '金額',
        '竣工日', '着手日', '現場代理人a', '監理技術者', '主任技術者',
        '現場担当者１', '現場担当者２', '工事場所', 'JV比率', '特記工法'
    ]
    try:
        df_k = conn.read(worksheet=KOUJI_SHEET, ttl=0)
    except Exception as e:
        st.error(f"工事データの読み込みに失敗しました: {e}")
        df_k = pd.DataFrame()

    if df_k.empty:
        df_k = pd.DataFrame(columns=core_k_cols)
    for c in core_k_cols:
        if c not in df_k.columns:
            df_k[c] = ""
    for col in df_k.columns:
        if "日" in col:
            df_k[col] = pd.to_datetime(df_k[col], errors="coerce")

    # --- 技術者データ ---
    core_e_cols = [
        '氏名', '保有資格', '資格', '在籍状況', '技術者ID',
        '監理技術者資格者証番号', '交付日', '有効期限日',
        '監理技術者講習修了年月日', '最新更新日'
    ]
    try:
        df_e = conn.read(worksheet=ENGINEER_SHEET, ttl=0)
    except Exception as e:
        st.error(f"技術者データの読み込みに失敗しました: {e}")
        df_e = pd.DataFrame()

    if df_e.empty:
        df_e = pd.DataFrame(columns=core_e_cols)
    for col in core_e_cols:
        if col not in df_e.columns:
            if col == '在籍状況':
                df_e[col] = True
            else:
                df_e[col] = ""
    for col in df_e.columns:
        if "日" in col:
            df_e[col] = pd.to_datetime(df_e[col], errors="coerce")
    if "技術者ID" in df_e.columns:
        df_e["技術者ID"] = df_e["技術者ID"].fillna("").astype(str)
        df_e["技術者ID"] = df_e["技術者ID"].str.replace(r"\.0$", "", regex=True)
        df_e["技術者ID"] = df_e["技術者ID"].replace("nan", "")
    if "在籍状況" in df_e.columns:
        df_e["在籍状況"] = df_e["在籍状況"].fillna(True).astype(bool)

    return df_k, df_e

# =========================
# 保存用関数（シートごとに分割）
# =========================
def save_kouji(df):
    """工事データのみを更新する"""
    try:
        conn.update(worksheet=KOUJI_SHEET, data=df)
        return True
    except Exception as e:
        st.error(f"工事データの保存中にエラーが発生しました: {e}")
        return False

def save_engineer(df):
    """技術者データのみを更新する"""
    try:
        conn.update(worksheet=ENGINEER_SHEET, data=df)
        return True
    except Exception as e:
        st.error(f"技術者データの保存中にエラーが発生しました: {e}")
        return False

# データ読み込み
df_kouji_raw, df_eng_raw = load_data()

# =========================
# 項目追加・削除のモーダルダイアログ
# =========================
@st.dialog("➕ 項目（列）を追加")
def dialog_add_column(target):
    new_col = st.text_input("追加する項目名を入力してください", placeholder="例: 備考、電話番号")
    if st.button("追加する", type="primary", use_container_width=True):
        if not new_col or not new_col.strip():
            st.warning("項目名を入力してください。")
            return
        new_col = new_col.strip()
        key = f"extra_cols_{target}"
        if key not in st.session_state:
            st.session_state[key] = []
        src_df = df_kouji_raw if target == "kouji" else df_eng_raw
        if new_col in src_df.columns or new_col in st.session_state[key]:
            st.warning(f"「{new_col}」は既に存在します。")
        else:
            st.session_state[key].append(new_col)
            st.rerun()

@st.dialog("🗑 項目（列）を削除")
def dialog_del_column(target):
    src_df = df_kouji_raw if target == "kouji" else df_eng_raw
    # extra_colsも含める
    extra_key = f"extra_cols_{target}"
    extra_cols = st.session_state.get(extra_key, [])
    all_cols = src_df.columns.tolist()
    for c in extra_cols:
        if c not in all_cols:
            all_cols.append(c)
    if not all_cols:
        st.warning("データがありません。")
        return
    sel_col = st.selectbox("削除する項目を選択", all_cols, index=None, placeholder="削除する項目を選んでください")
    if sel_col:
        st.warning(f"⚠️ 「**{sel_col}**」を削除します。「保存」ボタンを押すとスプレッドシートからも完全に削除されます。元に戻せません。")
        if st.button(f"🗑 「{sel_col}」を削除する", type="primary", use_container_width=True):
            key = f"del_cols_{target}"
            if key not in st.session_state:
                st.session_state[key] = []
            if sel_col not in st.session_state[key]:
                st.session_state[key].append(sel_col)
            # extra_colsからも除去
            if sel_col in st.session_state.get(extra_key, []):
                st.session_state[extra_key].remove(sel_col)
            st.rerun()
    else:
        st.info("削除する項目を選択してください。")

# =========================
# タブ画面構成
# =========================
tab1, tab2, tab3 = st.tabs(["🔍 実績検索", "✏️ 工事データ登録・編集", "👤 技術者登録・編集"])

# --- タブ1: 検索（技術者ベース × 実績条件） ---
with tab1:
    df_search = df_kouji_raw.copy()

    # 1. 金額の数値化処理
    price_cols = [c for c in df_search.columns if "金額" in c]
    if price_cols:
        target_col = price_cols[0]
        df_search["search_price"] = df_search[target_col].apply(process_price_data)
    else:
        df_search["search_price"] = 0

    # 2. 年の数値化処理
    if "竣工日" in df_search.columns:
        df_search["竣工日_dt"] = pd.to_datetime(df_search["竣工日"], errors="coerce")
        df_search["竣工年_val"] = df_search["竣工日_dt"].dt.year.fillna(0).astype(int)
    else:
        df_search["竣工年_val"] = 0

    # 全文検索用カラム作成
    def combine_all_columns(row):
        text = ""
        for val in row.values:
            if pd.notnull(val):
                text += normalize_text(val) + " "
        return text

    if not df_search.empty:
        df_search['full_text_search'] = df_search.apply(combine_all_columns, axis=1)
    else:
        df_search['full_text_search'] = ""

    # ========================================================
    # 在籍技術者リストの準備（部分一致検索用）
    # ========================================================
    engineer_map = {}
    active_engineer_list = []

    if not df_eng_raw.empty:
        active_engineers_df = df_eng_raw[df_eng_raw['在籍状況'] == True]
    else:
        active_engineers_df = pd.DataFrame()

    active_names = []
    active_quals = []
    name_col = '氏名'

    if not active_engineers_df.empty:
        if '氏名' in active_engineers_df.columns:
            name_col = '氏名'
        else:
            name_col = active_engineers_df.columns[0]

        raw_names = active_engineers_df[name_col].dropna().astype(str).unique().tolist()
        active_names = sorted(raw_names)

        qual_set = set()
        if '保有資格' in active_engineers_df.columns:
            raw_vals = active_engineers_df['保有資格'].dropna().astype(str)
            for v in raw_vals:
                splits = re.split(r'[\s\u3000,、]+', v.strip())
                for s in splits:
                    if s:
                        qual_set.add(s)
        active_quals = sorted(list(qual_set))

        qual_col = '保有資格' if '保有資格' in active_engineers_df.columns else '資格'
        for index, row in active_engineers_df.iterrows():
            if pd.notnull(row[name_col]):
                nm = row[name_col]
                clean_key = clean_string_for_match(nm)
                raw_qual = ""
                if qual_col in row and pd.notnull(row[qual_col]):
                    val = row[qual_col]
                    if str(val).lower() != 'nan' and str(val).strip() != "":
                        raw_qual = str(val).strip().replace("\n", " ")
                engineer_map[clean_key] = raw_qual
                active_engineer_list.append((clean_key, nm))

        active_engineer_list.sort(key=lambda x: len(x[0]), reverse=True)

    # ========================================================
    # サイドバー UI
    # ========================================================
    st.sidebar.header("🔍 検索条件")

    if not df_search.empty:
        min_p = int(df_search['search_price'].min())
        max_p = int(df_search['search_price'].max())
        MAX_SAFE_PRICE = 1_000_000_000_000
        if max_p > MAX_SAFE_PRICE:
            max_p = MAX_SAFE_PRICE
        if max_p <= min_p:
            max_p = min_p + 10000000
        kouji_types = df_search['工種名'].dropna().unique().tolist() if '工種名' in df_search.columns else []
        raw_years = df_search['竣工年_val'].unique()
        years = sorted([int(y) for y in raw_years if y > 0], reverse=True)
        if not years:
            years = [2025]
    else:
        min_p, max_p = 0, 10000000
        kouji_types = []
        years = [2025]

    # Session State
    if 'price_key' not in st.session_state:
        st.session_state['price_key'] = (min_p, max_p)
    if 'type_key' not in st.session_state:
        st.session_state['type_key'] = []
    if 'start_year_key' not in st.session_state:
        st.session_state['start_year_key'] = years[-1] if years else 2000
    if 'end_year_key' not in st.session_state:
        st.session_state['end_year_key'] = years[0] if years else 2025
    if 'target_names_key' not in st.session_state:
        st.session_state['target_names_key'] = []
    if 'target_quals_key' not in st.session_state:
        st.session_state['target_quals_key'] = []
    if 'role_key' not in st.session_state:
        st.session_state['role_key'] = []
    if 'keyword_count' not in st.session_state:
        st.session_state['keyword_count'] = 1
    # 数量キーワード検索用の Session State
    if 'qty_keyword_count' not in st.session_state:
        st.session_state['qty_keyword_count'] = 1

    def clear_form():
        st.session_state['price_key'] = (min_p, max_p)
        st.session_state['type_key'] = []
        st.session_state['start_year_key'] = years[-1] if years else 2000
        st.session_state['end_year_key'] = years[0] if years else 2025
        st.session_state['target_names_key'] = []
        st.session_state['target_quals_key'] = []
        st.session_state['role_key'] = []
        st.session_state['keyword_count'] = 1
        for k in list(st.session_state.keys()):
            if k.startswith('kw_input_'):
                st.session_state[k] = ""
        # 数量キーワード検索もリセット
        st.session_state['qty_keyword_count'] = 1
        for k in list(st.session_state.keys()):
            if k.startswith('qty_kw_select_'):
                st.session_state[k] = 0  # selectboxのインデックスを0（未選択）に
            elif k.startswith('qty_kw_value_'):
                st.session_state[k] = 0.0
            elif k.startswith('qty_kw_unit_placeholder_'):
                pass  # そのまま

    if st.sidebar.button("🔄 データの再読み込み"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.button("条件リセット", on_click=clear_form)

    # キーワード検索セクション（フォーム外に配置）
    st.sidebar.markdown("### 🔍 キーワード検索 (AND条件)")

    def remove_keyword_row(row_idx):
        """キーワード行を削除する"""
        count = st.session_state.get('keyword_count', 1)
        if count <= 1:
            st.session_state[f'kw_input_{row_idx}'] = ""
            return
        for j in range(row_idx, count - 1):
            next_val = st.session_state.get(f'kw_input_{j+1}', "")
            st.session_state[f'kw_input_{j}'] = next_val
        last = count - 1
        k = f'kw_input_{last}'
        if k in st.session_state:
            del st.session_state[k]
        st.session_state['keyword_count'] = count - 1

    keywords = []
    kw_count = st.session_state.get('keyword_count', 1)
    for i in range(kw_count):
        val = st.sidebar.text_input(f"キーワード {i+1}", key=f"kw_input_{i}")
        if val:
            keywords.append(val)
        if kw_count > 1:
            st.sidebar.button(f"✖ キーワード {i+1} を削除", key=f"del_kw_{i}", on_click=remove_keyword_row, args=(i,))
    st.sidebar.button("+ キーワード欄追加", on_click=lambda: st.session_state.update({'keyword_count': st.session_state.get('keyword_count', 1) + 1}), key="add_keyword_btn")

    # ========================================================
    # 数量キーワード検索セクション（フォーム外に配置）
    # ========================================================
    st.sidebar.markdown("### 📋 数量条件検索")
    st.sidebar.caption("工種を選択し、数量の下限値を指定して検索できます")

    qty_keyword_list = list(QUANTITY_KEYWORDS.keys())
    qty_options = ["（選択してください）"] + qty_keyword_list

    def remove_qty_row(row_idx):
        """数量条件行を削除する"""
        count = st.session_state.get('qty_keyword_count', 1)
        if count <= 1:
            # 最後の1行は削除せずリセットのみ
            st.session_state[f'qty_kw_select_{row_idx}'] = 0
            if f'qty_kw_value_{row_idx}' in st.session_state:
                st.session_state[f'qty_kw_value_{row_idx}'] = 0.0
            return
        # 削除対象以降の行を前にずらす
        for j in range(row_idx, count - 1):
            next_sel = st.session_state.get(f'qty_kw_select_{j+1}', 0)
            next_val = st.session_state.get(f'qty_kw_value_{j+1}', 0.0)
            st.session_state[f'qty_kw_select_{j}'] = next_sel
            st.session_state[f'qty_kw_value_{j}'] = next_val
        # 最後の行のキーを削除
        last = count - 1
        for prefix in ['qty_kw_select_', 'qty_kw_value_', 'qty_kw_unit_placeholder_']:
            k = f'{prefix}{last}'
            if k in st.session_state:
                del st.session_state[k]
        st.session_state['qty_keyword_count'] = count - 1

    for i in range(st.session_state.get('qty_keyword_count', 1)):
        qty_count = st.session_state.get('qty_keyword_count', 1)
        col_kw, col_val = st.sidebar.columns([3, 2])

        with col_kw:
            selected_kw_idx = st.selectbox(
                f"工種 {i+1}",
                options=range(len(qty_options)),
                format_func=lambda x: qty_options[x],
                key=f"qty_kw_select_{i}"
            )
            selected_kw = qty_options[selected_kw_idx] if selected_kw_idx < len(qty_options) else "（選択してください）"

        with col_val:
            if selected_kw and selected_kw != "（選択してください）":
                unit, step_size = QUANTITY_KEYWORDS.get(selected_kw, ("", 1))
                qty_val = st.number_input(
                    f"{unit} 以上",
                    min_value=0.0,
                    value=0.0,
                    step=float(step_size),
                    key=f"qty_kw_value_{i}"
                )
            else:
                st.text_input("単位", value="—", disabled=True, key=f"qty_kw_unit_placeholder_{i}")

        if qty_count > 1:
            st.sidebar.button(f"✖ 工種 {i+1} を削除", key=f"del_qty_{i}", on_click=remove_qty_row, args=(i,))

    st.sidebar.button(
        "+ 数量条件を追加",
        on_click=lambda: st.session_state.update({
            'qty_keyword_count': st.session_state.get('qty_keyword_count', 1) + 1
        }),
        key="add_qty_keyword_btn"
    )

    with st.sidebar.form("search_form"):
        step_val = 1000000
        if max_p - min_p < step_val:
            step_val = max(1, int((max_p - min_p) / 10))
        price_range = st.slider("金額 (円以上)", min_p, max_p, step=step_val, key='price_key')

        sel_types = st.multiselect("工種", kouji_types, key='type_key')

        st.markdown("### 📅 竣工年で絞り込み")
        c1, c2 = st.columns(2)
        with c1:
            start_year = st.selectbox("開始年", years, key='start_year_key')
        with c2:
            end_year = st.selectbox("終了年", years, key='end_year_key')

        st.markdown("### 👤 技術者名で検索 (複数可)")
        target_names = st.multiselect("指名検索", active_names, key='target_names_key')

        st.markdown("### 🎫 保有資格で検索")
        target_quals = st.multiselect("資格名を選択", active_quals, key='target_quals_key')

        role_cols = ['現場代理人', '監理技術者', '主任技術者', '現場担当者１', '現場担当者２']
        avail_roles = [r for r in role_cols if r in df_search.columns]
        st.markdown("### 詳細フィルター")
        sel_roles = st.multiselect("対象役職", avail_roles, key='role_key')

        search_btn = st.form_submit_button("検索")

    # ========================================================
    # 検索ロジックと結果表示
    # ========================================================
    if df_search.empty:
        st.warning("データがありません。")
    else:
        # 検索時にキーワードを再取得（セッションステートから）
        search_keywords = []
        for i in range(st.session_state.get('keyword_count', 1)):
            kw_val = st.session_state.get(f'kw_input_{i}', '')
            if kw_val:
                search_keywords.append(kw_val)

        # 数量キーワード条件を取得
        qty_conditions = []
        for i in range(st.session_state.get('qty_keyword_count', 1)):
            sel_kw_idx = st.session_state.get(f'qty_kw_select_{i}', 0)
            if isinstance(sel_kw_idx, int) and sel_kw_idx > 0 and sel_kw_idx < len(qty_options):
                sel_kw = qty_options[sel_kw_idx]
                min_val = st.session_state.get(f'qty_kw_value_{i}', 0.0)
                if min_val > 0:
                    qty_conditions.append((sel_kw, min_val))

        # 1. データの絞り込み
        df_res = df_search[
            (df_search['search_price'] >= price_range[0]) &
            (df_search['search_price'] <= price_range[1])
        ]

        if sel_types:
            df_res = df_res[df_res['工種名'].isin(sel_types)]

        if '竣工年_val' in df_res.columns:
            df_res = df_res[(df_res['竣工年_val'] >= start_year) & (df_res['竣工年_val'] <= end_year)]

        # 数量キーワードによるフィルタリング
        overview_col = '工事概要（主な工種、規格、数量）'
        if qty_conditions and overview_col in df_res.columns:
            for kw, min_val in qty_conditions:
                df_res = df_res[
                    df_res[overview_col].apply(
                        lambda x: extract_quantity_from_text(x, kw) >= min_val
                    )
                ]

        # 2. 検索対象技術者の決定
        search_target_list = []

        requested_names = list(target_names)
        if target_quals and not active_engineers_df.empty:
            if '保有資格' in active_engineers_df.columns:
                def check_qual_contain(val):
                    if pd.isnull(val):
                        return False
                    val_str = str(val)
                    norm_val = normalize_text(val_str)
                    tokens = set(re.split(r'[\s\u3000]+', norm_val.strip()))
                    norm_targets = [normalize_text(t) for t in target_quals]
                    return not tokens.isdisjoint(norm_targets)
                matched_engs = active_engineers_df[active_engineers_df['保有資格'].apply(check_qual_contain)]
                if not matched_engs.empty:
                    requested_names.extend(matched_engs[name_col].dropna().astype(str).tolist())
        requested_names = list(set(requested_names))

        if requested_names:
            for nm in requested_names:
                search_target_list.append((clean_string_for_match(nm), nm))
        else:
            search_target_list = active_engineer_list

        # 3. 検索実行（名前でフィルタリング）
        if requested_names:
            target_cleans = [t[0] for t in search_target_list]
            def check_row_contains_target(row):
                for r in avail_roles:
                    val = row.get(r)
                    if pd.notnull(val):
                        c_val = clean_string_for_match(val)
                        for t_clean in target_cleans:
                            if t_clean in c_val:
                                return True
                return False
            df_res = df_res[df_res.apply(check_row_contains_target, axis=1)]

        # キーワードフィルタ
        if search_keywords:
            for k in search_keywords:
                k_norm = normalize_text(k)
                df_res = df_res[df_res['full_text_search'].str.contains(k_norm, na=False)]

        # --- 集計処理 ---
        results = {}
        search_roles_final = sel_roles if sel_roles else avail_roles
        system_cols = ['search_price', 'full_text_search', '竣工日_dt', '竣工年_val']

        for idx, row in df_res.iterrows():
            for role in search_roles_final:
                raw_val = row.get(role)
                if pd.isnull(raw_val) or str(raw_val).strip() == "":
                    continue
                cell_clean = clean_string_for_match(raw_val)

                for eng_clean, eng_display in search_target_list:
                    if eng_clean in cell_clean:
                        is_display_target = True
                        if search_keywords:
                            for k in search_keywords:
                                if normalize_text(k) not in row['full_text_search']:
                                    is_display_target = False
                                    break
                        if is_display_target:
                            if eng_display not in results:
                                p_qual = engineer_map.get(eng_clean, "")
                                results[eng_display] = {"qualification": p_qual, "projects": []}
                            item = row.to_dict()
                            item['役割'] = role
                            results[eng_display]["projects"].append(item)

        # 数量条件の表示
        if qty_conditions:
            cond_texts = []
            for kw, min_val in qty_conditions:
                unit, _ = QUANTITY_KEYWORDS.get(kw, ("", 1))
                cond_texts.append(f"**{kw}** {min_val:,.0f}{unit}以上")
            st.info("📋 数量条件: " + " ／ ".join(cond_texts))

        st.subheader(f"検索結果: {len(results)} 名")
        st.write("---")

        for name in sorted(results.keys()):
            data = results[name]
            qual_display = data['qualification']
            if qual_display and qual_display.lower() != 'nan':
                st.markdown(f"### 👤 {name} 【{qual_display}】")
            else:
                st.markdown(f"### 👤 {name}")

            p_df = pd.DataFrame(data['projects'])
            if not p_df.empty:
                if 'search_price' in p_df.columns:
                    p_df = p_df.sort_values('search_price', ascending=False)

                all_cols = p_df.columns.tolist()
                orig_csv_cols = [c for c in df_kouji_raw.columns if c not in system_cols]
                final_order = ['役割']
                for c in orig_csv_cols:
                    if c in p_df.columns:
                        final_order.append(c)
                for c in all_cols:
                    if c not in final_order and c not in system_cols:
                        final_order.append(c)

                display_df = p_df[final_order].copy()
                for col in display_df.columns:
                    if '日' in col:
                        display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y/%m/%d').fillna('')

                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.markdown("---")

# --- タブ2: 工事登録 ---
with tab2:
    st.header("✏️ 工事実績データの管理")
    st.info("データの追加・修正を行い「保存」を押してください。（保存ボタンを押すまで反映されません）")

    # --- 項目（列）追加・削除 UI ---
    if "extra_cols_kouji" not in st.session_state:
        st.session_state.extra_cols_kouji = []

    # 追加列・削除列をボタンの前に反映（ダイアログが正しいリストを表示するため）
    for col in st.session_state.extra_cols_kouji:
        if col not in df_kouji_raw.columns:
            df_kouji_raw[col] = None
    if "del_cols_kouji" in st.session_state and st.session_state.del_cols_kouji:
        st.warning(f"⚠️ 削除予定の項目: {', '.join(st.session_state.del_cols_kouji)}（保存で確定・取り消すにはページを再読み込み）")
        for col in st.session_state.del_cols_kouji:
            if col in df_kouji_raw.columns:
                df_kouji_raw = df_kouji_raw.drop(columns=[col])

    btn_k1, btn_k2, btn_k3 = st.columns([1, 1, 3])
    with btn_k1:
        if st.button("➕ 項目を追加", key="btn_add_col_kouji"):
            dialog_add_column("kouji")
    with btn_k2:
        if st.button("🗑 項目を削除", key="btn_del_col_kouji"):
            dialog_del_column("kouji")

    k_col_cfg = {}
    if not df_kouji_raw.empty:
        for c in df_kouji_raw.columns:
            if "日" in c:
                k_col_cfg[c] = st.column_config.DateColumn(c, format="YYYY/MM/DD")

    with st.form("kouji_form"):
        if not df_kouji_raw.empty:
            edited_kouji = st.data_editor(
                df_kouji_raw, num_rows="dynamic",
                use_container_width=True, column_config=k_col_cfg,
                key="editor_kouji"
            )
        else:
            st.warning("工事データが空です。新規登録してください。")
            edited_kouji = pd.DataFrame()

        submit_btn = st.form_submit_button("💾 工事データを上書き保存", type="primary")

    if submit_btn:
        if not edited_kouji.empty:
            if save_kouji(edited_kouji):
                st.success(f"シート「{KOUJI_SHEET}」に上書き保存しました！")
                st.session_state.extra_cols_kouji = []
                st.session_state.del_cols_kouji = []
                st.cache_data.clear()
                st.rerun()

# --- タブ3: 技術者登録 ---
with tab3:
    st.header("👤 技術者情報の管理")
    st.info("技術者の追加・修正を行い「保存」を押してください。（保存ボタンを押すまで反映されません）")

    # --- 項目（列）追加・削除 UI ---
    if "extra_cols_eng" not in st.session_state:
        st.session_state.extra_cols_eng = []

    # 追加列・削除列をボタンの前に反映
    for col in st.session_state.extra_cols_eng:
        if col not in df_eng_raw.columns:
            df_eng_raw[col] = None
    if "del_cols_eng" in st.session_state and st.session_state.del_cols_eng:
        st.warning(f"⚠️ 削除予定の項目: {', '.join(st.session_state.del_cols_eng)}（保存で確定・取り消すにはページを再読み込み）")
        for col in st.session_state.del_cols_eng:
            if col in df_eng_raw.columns:
                df_eng_raw = df_eng_raw.drop(columns=[col])

    btn_e1, btn_e2, btn_e3 = st.columns([1, 1, 3])
    with btn_e1:
        if st.button("➕ 項目を追加", key="btn_add_col_eng"):
            dialog_add_column("eng")
    with btn_e2:
        if st.button("🗑 項目を削除", key="btn_del_col_eng"):
            dialog_del_column("eng")

    e_col_cfg = {
        "在籍状況": st.column_config.CheckboxColumn("在籍", default=True),
        "技術者ID": st.column_config.TextColumn("技術者ID", width="medium", required=True),
        "保有資格": st.column_config.TextColumn("保有資格", width="large"),
    }
    if not df_eng_raw.empty:
        for c in df_eng_raw.columns:
            if "日" in c:
                e_col_cfg[c] = st.column_config.DateColumn(c, format="YYYY/MM/DD")

    with st.form("engineer_form"):
        if not df_eng_raw.empty:
            hide_cols = ['資格', '資格名称']
            all_cols = df_eng_raw.columns.tolist()
            display_cols = [c for c in all_cols if c not in hide_cols]
            edited_eng = st.data_editor(
                df_eng_raw, column_order=display_cols,
                num_rows="dynamic", column_config=e_col_cfg,
                use_container_width=True, key="editor_eng"
            )
        else:
            st.warning("技術者データが空です。")
            edited_eng = pd.DataFrame()

        submit_btn_eng = st.form_submit_button("💾 技術者データを上書き保存", type="primary")

    if submit_btn_eng:
        if not edited_eng.empty:
            if save_engineer(edited_eng):
                st.success(f"シート「{ENGINEER_SHEET}」に上書き保存しました！")
                st.session_state.extra_cols_eng = []
                st.session_state.del_cols_eng = []
                st.cache_data.clear()
                st.rerun()
