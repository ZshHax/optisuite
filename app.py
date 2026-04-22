import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import linprog, minimize
import math
import time
import random

# --- 1. CONFIGURATION & CSS ---
st.set_page_config(
    page_title="Intelligent Optimization Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Global Styles */
    .stApp {background-color: #f8f9fa;}

    /* Typography */
    h1, h2, h3 {font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #2c3e50; font-weight: 600;}
    p, li, label, div {font-family: 'Segoe UI', Tahoma, sans-serif; color: #34495e;}

    /* Cards/Containers */
    div.css-1r6slb0 {background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}

    /* Sidebar */
    section[data-testid="stSidebar"] {background-color: #ffffff; border-right: 1px solid #e9ecef;}

    /* Buttons */
    .stButton>button {
        width: 100%; border-radius: 6px; font-weight: 500; border: none;
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); color: white;
        padding: 0.5rem 1rem; transition: all 0.2s;
    }
    .stButton>button:hover {transform: translateY(-2px); box-shadow: 0 4px 12px rgba(41, 128, 185, 0.3);}

    /* Custom Elements */
    .metric-card {background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .success-box {padding: 15px; background-color: #d4edda; color: #155724; border-radius: 5px; border: 1px solid #c3e6cb;}
    </style>
""", unsafe_allow_html=True)


# --- 2. MATH CORE ---
def safe_eval(formula, x, y, t=0):
    safe_dict = {
        "x": x, "y": y, "t": t, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "pi": np.pi, "sqrt": np.sqrt, "exp": np.exp, "abs": np.abs, "log": np.log, "e": np.e
    }
    try:
        return eval(formula, {"__builtins__": None}, safe_dict)
    except:
        return 0


# --- 3. LEVEL GENERATOR ---
def get_level_data(n):
    # Ручные сценарии (1-5)
    if n == 1: return {"type": "1d", "title": "Урок 1: Введение", "desc": "Найдите вершину параболы.",
                       "formula": "-1*(x-2)**2 + 8", "target": 2.0, "tol": 0.5,
                       "hint": "Двигайте ползунок к пику графика."}

    if n == 2: return {"type": "quiz", "title": "Урок 2: Теория", "desc": "Базовые понятия.",
                       "question": "Что показывает целевая функция?",
                       "opts": ["Критерий эффективности решения", "Систему ограничений", "Скорость вычислений"],
                       "ans": "Критерий эффективности решения"}

    if n == 3: return {"type": "1d", "title": "Урок 3: Модуль", "desc": "Найдите минимум V-образной функции.",
                       "formula": "abs(x-4)", "target": 4.0, "tol": 0.3,
                       "hint": "Минимум там, где функция касается нуля."}

    if n == 4: return {"type": "choice", "title": "Урок 4: Градиент",
                       "desc": "Вы на склоне Y=X^2 в точке X=5. Куда спускаться?",
                       "opts": ["Влево (к нулю)", "Вправо (в бесконечность)"], "ans": "Влево (к нулю)",
                       "hint": "Нам нужен минимум (дно)."}

    if n == 5: return {"type": "3d", "title": "Урок 5: 3D Пространство", "desc": "Найдите дно чаши.",
                       "formula": "(x-1)**2 + (y+1)**2", "target": [1, -1], "tol": 1.2,
                       "hint": "Ищите самую темную область."}

    # Генерация (6-50)
    task_type = random.choice(["1d", "quiz", "3d", "choice"])

    if n % 10 == 0:  # Boss Level
        tx, ty = random.randint(-4, 4), random.randint(-4, 4)
        return {"type": "3d", "title": f"ЭКЗАМЕН: Уровень {n}", "desc": "Сложная функция. Высокая точность.",
                "formula": f"10 - 5*exp(-((x-{tx})**2 + (y-{ty})**2))", "target": [tx, ty], "tol": 0.5,
                "hint": f"Центр аномалии около ({tx}, {ty})"}

    if task_type == "1d":
        tgt = random.randint(-8, 8)
        return {"type": "1d", "title": f"Уровень {n}: Практика 1D", "desc": "Найдите экстремум.",
                "formula": f"-(x-{tgt})**2", "target": float(tgt), "tol": 0.5, "hint": "Ищите вершину."}

    elif task_type == "quiz":
        q_db = [
            ("Какой метод не требует производных?", "Метод Нелдера-Мида", ["Градиентный спуск", "Метод Ньютона"]),
            ("В задаче ЛП оптимум всегда...", "На границе области", ["В центре области", "Вне области"]),
            ("Глобальный минимум - это...", "Самая низкая точка всей функции", ["Любая впадина", "Точка перегиба"]),
            ("Что такое 'Constraints'?", "Ограничения", ["Переменные", "Целевая функция"])
        ]
        q, a, dist = random.choice(q_db)
        opts = dist + [a]
        random.shuffle(opts)
        return {"type": "quiz", "title": f"Уровень {n}: Блиц", "desc": "Проверка знаний.", "question": q, "opts": opts,
                "ans": a}

    elif task_type == "choice":
        start = random.choice([-5, 5])
        ans = "Вправо" if start < 0 else "Влево"
        return {"type": "choice", "title": f"Уровень {n}: Направление",
                "desc": f"Склон параболы, X={start}. Куда вниз?", "opts": ["Влево", "Вправо"], "ans": ans,
                "hint": "К нулю."}

    elif task_type == "3d":
        tx, ty = random.randint(-6, 6), random.randint(-6, 6)
        return {"type": "3d", "title": f"Уровень {n}: 3D Сканирование", "desc": "Найдите минимум.",
                "formula": f"(x-{tx})**2 + (y-{ty})**2", "target": [tx, ty], "tol": 1.5, "hint": f"Сектор {tx}, {ty}"}


# --- 4. UI LOGIC ---
if 'level' not in st.session_state: st.session_state.level = 1
if 'xp' not in st.session_state: st.session_state.xp = 0

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
    st.title("OptiSuite Pro")
    st.caption("v.2026.1 | Build 8402")
    st.markdown("---")

    menu = st.radio("Навигация",
                    ["🎓 Академия", "📐 Линейная оптимизация", "🏔 Нелинейная оптимизация", "🌀 4D Анализ", "📘 Фундаментальная теория"])

    if menu == "🎓 Академия":
        st.markdown("---")
        with st.expander("🛠 Панель преподавателя"):
            lvl_sel = st.number_input("Перейти к уровню", 1, 50, st.session_state.level)
            if st.button("Перейти"): st.session_state.level = lvl_sel; st.rerun()
            if st.button("Сброс курса"): st.session_state.level = 1; st.session_state.xp = 0; st.rerun()
        st.metric("Уровень", st.session_state.level)
        st.metric("XP", st.session_state.xp)

# --- MODULE: ACADEMY ---
if menu == "🎓 Академия":
    if st.session_state.level > 50:
        st.balloons()
        st.success("🎉 КУРС ЗАВЕРШЕН! Вы готовы к защите диплома.")
    else:
        data = get_level_data(st.session_state.level)

        st.subheader(data['title'])
        st.progress(st.session_state.level / 50)

        # Основной контейнер задачи
        with st.container():
            col_task, col_vis = st.columns([1, 2])

            with col_task:
                st.info(data['desc'])

                # --- ТИП: 1D SLIDER ---
                if data['type'] == "1d":
                    val = st.slider("Значение X", -10.0, 10.0, 0.0, 0.1)
                    if st.button("Проверить"):
                        if abs(val - data['target']) <= data['tol']:
                            st.balloons();
                            st.success("Верно! +50 XP");
                            time.sleep(1)
                            st.session_state.level += 1;
                            st.session_state.xp += 50;
                            st.rerun()
                        else:
                            st.error(f"Ошибка. {data['hint']}")

                # --- ТИП: QUIZ (ИСПРАВЛЕНО) ---
                elif data['type'] == "quiz":
                    st.markdown(f"### ❓ {data['question']}")  # ЯВНЫЙ ВЫВОД ВОПРОСА
                    ans = st.radio("Ваш ответ:", data['opts'], index=None)
                    if st.button("Ответить"):
                        if ans == data['ans']:
                            st.balloons();
                            st.success("Правильно! +20 XP");
                            time.sleep(1)
                            st.session_state.level += 1;
                            st.session_state.xp += 20;
                            st.rerun()
                        elif ans:
                            st.error("Неверно.")
                        else:
                            st.warning("Выберите вариант.")

                # --- ТИП: CHOICE ---
                elif data['type'] == "choice":
                    c1, c2 = st.columns(2)
                    if c1.button(data['opts'][0]):
                        if data['opts'][0] == data['ans']:
                            st.balloons();
                            st.session_state.level += 1;
                            st.rerun()
                        else:
                            st.error("Не туда")
                    if c2.button(data['opts'][1]):
                        if data['opts'][1] == data['ans']:
                            st.balloons();
                            st.session_state.level += 1;
                            st.rerun()
                        else:
                            st.error("Не туда")

                # --- ТИП: 3D SCAN ---
                elif data['type'] == "3d":
                    ux = st.number_input("X", -10.0, 10.0, 0.0)
                    uy = st.number_input("Y", -10.0, 10.0, 0.0)
                    if st.button("Сканировать"):
                        dist = np.sqrt((ux - data['target'][0]) ** 2 + (uy - data['target'][1]) ** 2)
                        if dist <= data['tol']:
                            st.balloons();
                            st.success("Цель найдена! +100 XP");
                            time.sleep(1)
                            st.session_state.level += 1;
                            st.session_state.xp += 100;
                            st.rerun()
                        else:
                            st.warning(f"Мимо. Дистанция: {dist:.1f}")

            with col_vis:
                # Визуализация зависит от типа
                if data['type'] in ["1d", "choice"]:
                    x = np.linspace(-10, 10, 100)
                    y = [safe_eval(data.get('formula', 'x**2'), i, 0) for i in x]
                    fig = px.line(x=x, y=y, title="График функции")
                    if data['type'] == '1d': fig.add_vline(x=val, line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                elif data['type'] == "3d":
                    x = np.linspace(-10, 10, 30)
                    y = np.linspace(-10, 10, 30)
                    X, Y = np.meshgrid(x, y)
                    Z = np.vectorize(lambda x, y: safe_eval(data['formula'], x, y))(X, Y)
                    fig = go.Figure(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8))
                    fig.add_trace(go.Scatter3d(x=[locals().get('ux', 0)], y=[locals().get('uy', 0)], z=[
                        safe_eval(data['formula'], locals().get('ux', 0), locals().get('uy', 0))], mode='markers',
                                               marker=dict(size=8, color='red')))
                    st.plotly_chart(fig, use_container_width=True)

# --- MODULE: LINEAR SOLVER ---
elif menu == "📐 Линейная оптимизация":
    st.header("📐 Simplex Solver")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Параметры")
        if st.button("⚡ Загрузить пример"):
            st.session_state.lp = pd.DataFrame(
                [{"a": 2.0, "b": 1.0, "limit": 10.0}, {"a": 1.0, "b": 3.0, "limit": 15.0}])

        coef_c1 = st.number_input("C1 (x1)", value=3.0)
        coef_c2 = st.number_input("C2 (x2)", value=4.0)
        goal = st.selectbox("Цель", ["Max", "Min"])

        df = st.data_editor(st.session_state.get('lp', pd.DataFrame([{"a": 1.0, "b": 1.0, "limit": 10.0}])),
                            num_rows="dynamic")
        solve_btn = st.button("Рассчитать")

    with c2:
        if solve_btn:
            try:
                c = [-coef_c1, -coef_c2] if goal == "Max" else [coef_c1, coef_c2]
                A = df[["a", "b"]].values;
                b = df["limit"].values
                res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method='highs')

                if res.success:
                    val = -res.fun if goal == "Max" else res.fun
                    st.success("✅ Оптимальное решение найдено")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Z", f"{val:.2f}");
                    k2.metric("X1", f"{res.x[0]:.2f}");
                    k3.metric("X2", f"{res.x[1]:.2f}")

                    fig = go.Figure()
                    xr = np.linspace(0, max(res.x[0] * 1.5, 10), 100)
                    for i, r in df.iterrows():
                        if r['b'] != 0:
                            fig.add_trace(go.Scatter(x=xr, y=(r['limit'] - r['a'] * xr) / r['b'], name=f"Огр {i + 1}"))
                        elif r['a'] != 0:
                            fig.add_vline(x=r['limit'] / r['a'], line_dash="dash")
                    fig.add_trace(
                        go.Scatter(x=[res.x[0]], y=[res.x[1]], mode='markers', marker=dict(size=15, color='red'),
                                   name="Оптимум"))
                    fig.update_layout(height=500, xaxis_title="X1", yaxis_title="X2")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Решение не найдено.")
            except Exception as e:
                st.error(f"Ошибка данных: {e}")

# --- MODULE: NON-LINEAR SOLVER ---
elif menu == "🏔 Нелинейная оптимизация":
    st.header("🏔 Лаборатория Анализа")
    c1, c2 = st.columns([1, 2])
    with c1:
        func = st.text_input("f(x, y) =", "(x-2)**2 + (y-2)**2")
        with st.expander("Библиотека функций"):
            if st.button("Розенброк"): st.code("(1-x)**2 + 100*(y-x**2)**2")
            if st.button("Химмельблау"): st.code("(x**2+y-11)**2 + (x+y**2-7)**2")
        met = st.selectbox("Метод", ["Nelder-Mead", "BFGS", "Powell"])
        sx = st.number_input("Start X", -10.0, 10.0, -5.0)
        sy = st.number_input("Start Y", -10.0, 10.0, 5.0)
        run = st.button("Запуск")

    with c2:
        x = np.linspace(-6, 6, 50);
        y = np.linspace(-6, 6, 50);
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(lambda x, y: safe_eval(func, x, y))(X, Y)
        fig = go.Figure(go.Surface(z=Z, x=X, y=Y, colorscale='Earth', opacity=0.8))
        if run:
            px, py = [], []
            res = minimize(lambda a: safe_eval(func, a[0], a[1]), [sx, sy], method=met,
                           callback=lambda k: px.append(k[0]) or py.append(k[1]))
            if res.success:
                pz = [safe_eval(func, i, j) for i, j in zip(px, py)]
                fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers', line=dict(color='black', width=3)))
                fig.add_trace(go.Scatter3d(x=[res.x[0]], y=[res.x[1]], z=[res.fun], mode='markers',
                                           marker=dict(size=10, color='red')))
                st.success(f"Минимум: {res.fun:.4f}")
        fig.update_layout(height=600, margin=dict(t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

# --- MODULE: 4D ---
elif menu == "🌀 4D Анализ":
    st.header("🌀 Гиперпространство (Время T)")
    c1, c2 = st.columns([1, 3])
    with c1:
        f4 = st.text_input("f(x, y, t)", "sin(sqrt(x**2 + y**2) - t)")
        t = st.slider("Время T", 0.0, 10.0, 0.0, 0.1)
    with c2:
        x = np.linspace(-10, 10, 50);
        y = np.linspace(-10, 10, 50);
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(lambda x, y: safe_eval(f4, x, y, t))(X, Y)
        fig = go.Figure(go.Surface(z=Z, x=X, y=Y, colorscale='Spectral'))
        fig.update_layout(title=f"Срез T={t}", height=600)
        st.plotly_chart(fig, use_container_width=True)

# --- MODULE: KNOWLEDGE BASE ---
# ==========================================
# МОДУЛЬ 5: ФУНДАМЕНТАЛЬНАЯ ТЕОРИЯ
# ==========================================
elif menu == "📘 Фундаментальная теория":
    st.title("📘 Справочник по теории оптимизации")
    st.markdown("---")

    # Используем вкладки для структурирования большого объема знаний
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📏 Линейное программирование", "🏔 Нелинейное программирование", "🧮 Алгоритмы", "🏛 История и Личности"])

    with tab1:
        st.header("1. Линейное программирование (ЛП)")

        with st.expander("📌 Постановка задачи (Каноническая форма)", expanded=True):
            st.write(
                "Общая задача ЛП заключается в нахождении экстремума линейной целевой функции при линейных ограничениях.")
            st.markdown("**Математическая модель:**")
            st.latex(r"Z(\vec{x}) = \sum_{j=1}^{n} c_j x_j \to \max (\min)")
            st.markdown("При ограничениях:")
            st.latex(
                r"\begin{cases} \sum_{j=1}^{n} a_{ij} x_j = b_i, \quad i=1,\dots,m \\ x_j \geq 0, \quad j=1,\dots,n \end{cases}")
            st.caption(
                "Где $x_j$ — переменные, $c_j$ — коэффициенты целевой функции, $a_{ij}$ — технологические коэффициенты, $b_i$ — ресурсы.")

        with st.expander("💰 Экономический смысл"):
            st.write("""
            **Задача об использовании ресурсов:**
            Представьте, что завод производит $n$ видов продукции, используя $m$ видов ресурсов.
            - $c_j$ — прибыль от единицы $j$-й продукции.
            - $a_{ij}$ — сколько $i$-го ресурса нужно на единицу $j$-й продукции.
            - $b_i$ — запас $i$-го ресурса на складе.

            **Цель:** Составить план производства ($x_1, x_2...$), чтобы максимизировать прибыль, не превысив запасы склада.
            """)

        with st.expander("🔄 Двойственная задача"):
            st.write(
                "Каждой задаче ЛП (прямой) соответствует двойственная задача. Если прямая задача — это план производства, то двойственная — это оценка ресурсов.")
            st.latex(r"W(\vec{y}) = \sum_{i=1}^{m} b_i y_i \to \min")
            st.markdown("""
            **Теоремы двойственности:**
            1. Если одна задача имеет оптимум, то и вторая имеет оптимум, причем $Z_{max} = W_{min}$.
            2. Переменные $y_i$ называют **теневыми ценами** (shadow prices). Они показывают, насколько вырастет прибыль, если мы добавим единицу ресурса $b_i$.
            """)

    with tab2:
        st.header("2. Нелинейное программирование (НЛП)")

        st.info(
            "В отличие от ЛП, здесь целевая функция или ограничения могут быть кривыми (нелинейными). Это делает задачу намного сложнее, так как появляются локальные экстремумы.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Локальный vs Глобальный")
            st.write("Большинство методов находят только **локальный** минимум (дно ближайшей ямы).")
            # Можно вставить картинку (используем график plotly для иллюстрации)
            x = np.linspace(-2, 2, 50)
            fig = px.line(x=x, y=x ** 4 - x ** 2, title="Функция с двумя минимумами")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Пример: у функции два дна. Алгоритм может застрять в левом, не зная, что правое глубже.")

        with col2:
            st.subheader("Выпуклость (Convexity)")
            st.write("""
            Задача называется **выпуклой**, если:
            1. Целевая функция выпуклая (как чаша).
            2. Область допустимых решений выпуклая.

            **Важное свойство:** В выпуклых задачах любой локальный минимум является **глобальным**. Это "Рай для оптимизатора".
            """)

        st.markdown("---")
        st.subheader("Математический аппарат")
        st.write("Для поиска экстремума используются производные. В точке минимума градиент равен нулю:")
        st.latex(
            r"\nabla f(x) = \left( \frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n} \right) = 0")
        st.write(
            "Для проверки типа экстремума (минимум или максимум) используют **Матрицу Гессе** (вторые производные).")

    with tab3:
        st.header("🧮 Методы и Алгоритмы")
        st.write("Сравнительная таблица методов, реализованных в комплексе.")

        data = {
            "Метод": ["Симплекс-метод", "Градиентный спуск", "Нелдера-Мида", "BFGS", "Метод Ньютона"],
            "Тип задач": ["Линейные (ЛП)", "Нелинейные (НЛП)", "Нелинейные (НЛП)", "Нелинейные (НЛП)",
                          "Нелинейные (НЛП)"],
            "Нужны производные?": ["Нет", "Да (1-го порядка)", "Нет (Эвристика)", "Да (Квази-Ньютон)",
                                   "Да (2-го порядка)"],
            "Скорость": ["Средняя (точный)", "Зависит от шага", "Медленная", "Высокая", "Очень высокая"],
            "Особенности": ["Перебор вершин", "Может застрять", "Работает с 'шумными' функциями", "Стандарт индустрии",
                            "Требует много памяти"]
        }
        df = pd.DataFrame(data)
        st.table(df)

        st.markdown("### Подробности:")
        st.markdown("""
        * **Симплекс-метод:** Двигается по ребрам многогранника от вершины к вершине, пока не найдет лучшую.
        * **Нелдера-Мида (Амеба):** Представьте треугольник, который ползает по поверхности, растягивается и сжимается, нащупывая дно. Отлично подходит, если формулу функции сложно продифференцировать.
        """)

    with tab4:
        st.header("🏛 История развития")

        col_a, col_b = st.columns(2)

        with col_a:
            # Надежная ссылка на Канторовича (Wikimedia)
            st.image("kantorovich.jpg",
                     caption="Леонид Витальевич Канторович (1912–1986)")
            st.markdown("#### Советский приоритет")
            st.info("""
            **Отец линейного программирования.**
            В 1939 году к 27-летнему профессору ЛГУ Канторовичу обратились инженеры фанерного треста. Они не могли оптимально распределить станки для обработки древесины.

            Канторович решил задачу и понял, что открыл новый класс математических проблем. Он разработал **метод разрешающих множителей** (прообраз симплекс-метода).

            🏆 **Нобелевская премия по экономике (1975)** "за вклад в теорию оптимального распределения ресурсов".
            """)

        with col_b:
            # Надежная ссылка на Данцига (Wikimedia)
            st.image(
                "dancig.jpg",
                caption="Джордж Данциг (1914–2005)")
            st.markdown("#### Американский подход")
            st.warning("""
            **Создатель Симплекс-метода.**
            В 1947 году, работая в ВВС США, Данциг решал задачи планирования снабжения армии. Он формализовал задачу ЛП и придумал эффективный алгоритм её решения — **Симплекс-метод**.

            **Легенда о нерешаемых задачах:**
            Будучи аспирантом, Данциг опоздал на лекцию, увидел на доске две задачи и решил их как домашнее задание. Оказалось, это были две главные нерешенные проблемы статистики того времени, над которыми ученые бились годами. Этот случай вошел в историю (и в фильм "Умница Уилл Хантинг").
            """)