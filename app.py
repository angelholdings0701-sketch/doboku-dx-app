import streamlit as st
import pandas as pd
import unicodedata
import re
import os
from streamlit_gsheets import GSheetsConnection

# === ページ設定 ===
st.set_page_config(page_title="工事実績管理DB", layout="wide")

# ==========================================
# 🔐 セキュリティ設定（厳易パスワード認証）
# ==========================================
def check_password():
    """パスワード認証が通っていなければ入力を求め、停止する"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # 認証済みなら何もしない
    if st.session_state.authenticated:
        return True

    # 画面表示
    st.title("🔐 ログインが必要です")

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
        '竣工日', '着手日', '現場代理人', '監理技術者', '主任技術者',
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
# タブ画面構成
# =========================
tab1, tab2, tab3 = st.tabs(["🔍 実績検索", "✏️ 工事データ登録・編集", "🧑 技術者登録・編集"])

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
        for k in list(st.session_state.keys()):
            if k.startswith('kw_input_'):
                st.session_state[k] = ""
        # 数量キーワード検索もリセット
        st.session_state['qty_keyword_count'] = 1
        for k in list(st.session_state.keys()):
            if k.startswith('qty_kw_select_') or k.startswith('qty_kw_value_'):
                del st.session_state[k]

    if st.sidebar.button("🔄 データの再読み込み"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.button("条件リセット", on_click=clear_form)

    # キーワード検索セクション（フォーム外に配置）
    st.sidebar.markdown("### 🔍 キーワード検索 (AND条件)")
    keywords = []
    for i in range(st.session_state.get('keyword_count', 1)):
        val = st.sidebar.text_input(f"キーワード {i+1}", key=f"kw_input_{i}")
        if val:
            keywords.append(val)
        if i == 0:
            st.sidebar.button("+ キーワード枠を追加", on_click=lambda: st.session_state.update({'keyword_count': st.session_state.get('keyword_count', 1) + 1}), key="add_keyword_btn")

    # ========================================================
    # 数量キーワード検索セクション（フォーム外に配置）
    # ========================================================
    st.sidebar.markdown("### 📋 数量条件検索")
    st.sidebar.caption("工種を選択し、数量の下限値を指定して検索できます")

    qty_keyword_list = list(QUANTITY_KEYWORDS.keys())

    for i in range(st.session_state.get('qty_keyword_count', 1)):
        cols = st.sidebar.columns([3, 2])
        with cols[0]:
            selected_kw = st.selectbox(
                f"工種 {i+1}",
                options=["（選択してください）"] + qty_keyword_list,
                key=f"qty_kw_select_{i}"
            )
        with cols[1]:
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

        if i == 0:
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

        st.markdown("### 🧑 技術者名で検索 (複数可)")
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
            sel_kw = st.session_state.get(f'qty_kw_select_{i}', '（選択してください）')
            if sel_kw and sel_kw != '（選択してください）':
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
            st.info("📋 数量条件: " + " / ".join(cond_texts))

        st.subheader(f"検索結果: {len(results)} 名")
        st.write("---")

        for name in sorted(results.keys()):
            data = results[name]
            qual_display = data['qualification']
            if qual_display and qual_display.lower() != 'nan':
                st.markdown(f"### 🧑 {name} 【{qual_display}】")
            else:
                st.markdown(f"### 🧑 {name}")

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
                st.cache_data.clear()
                st.rerun()

# --- タブ3: 技術者登録 ---
with tab3:
    st.header("🧑 技術者情報の管理")
    st.info("技術者の追加・修正を行い「保存」を押してください。（保存ボタンを押すまで反映されません）")

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
                st.cache_data.clear()
                st.rerun()
