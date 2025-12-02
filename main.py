import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import warnings
import os

warnings.filterwarnings("ignore")

# --- КОНФІГУРАЦІЯ СТОРІНКИ ---
st.set_page_config(page_title="Прогноз цін (Сумська обл.)", layout="wide")

st.title("📊 Прогнозування цін на товари")
st.markdown("Аналіз та прогноз цін на основі завантаженого датасету.")

# --- 1. ЗАВАНТАЖЕННЯ ТА ОБРОБКА ДАНИХ ---
st.sidebar.header("Налаштування")

# Папка для шаблонів
DATA_FOLDER = 'datasets'

# Створення папки, якщо її немає
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Функція для парсингу специфічного формату дати 'YYYY-Mmm'
def parse_custom_date(date_str):
    try:
        y, m = date_str.split('-M')
        return pd.to_datetime(f"{y}-{m}-01")
    except Exception:
        return pd.NaT

@st.cache_data
def load_and_clean_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        # 1. Фільтруємо тільки ціни (ігноруємо індекси)
        if 'Показник' in df.columns:
            df = df[df['Показник'] == 'Середні споживчі ціни на товари (послуги)']
        
        # 2. Обробка дати
        if 'Період' in df.columns:
            df['Date'] = df['Період'].apply(parse_custom_date)
        else:
            st.error("Не знайдено колонку 'Період'")
            return None

        # 3. Перейменування колонок для зручності
        col_map = {
            'Тип товарів і послуг': 'Product_Name',
            'Значення cпостереження': 'Price',
            'Територіальний розріз': 'Region'
        }
        df.rename(columns=col_map, inplace=True)
        
        if 'Region' not in df.columns:
            df['Region'] = 'Unknown'

        # 4. Вибір потрібних колонок та сортування
        df = df[['Date', 'Region', 'Product_Name', 'Price']].sort_values('Date')
        
        # 5. Приведення ціни до числа
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Помилка обробки файлу: {e}")
        return None

# Логіка вибору джерела
data_source = st.sidebar.radio("Оберіть режим:", ["📁 Шаблони", "⬆️ Завантажити файл"])

df = None

if data_source == "📁 Шаблони":
    # Скануємо папку datasets
    available_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    
    if available_files:
        selected_file = st.sidebar.selectbox("Оберіть шаблон:", available_files)
        file_path = os.path.join(DATA_FOLDER, selected_file)
        df = load_and_clean_data(file_path)
    else:
        st.sidebar.warning(f"Папка '{DATA_FOLDER}' порожня! Додайте туди CSV файли.")

elif data_source == "⬆️ Завантажити файл":
    uploaded_file = st.sidebar.file_uploader("Завантажте CSV", type=["csv"])
    if uploaded_file is not None:
        df = load_and_clean_data(uploaded_file)

# Якщо дані не завантажені — зупиняємось
if df is None:
    st.info("Оберіть шаблон або завантажте файл для початку роботи.")
    st.stop()

# --- БЛОК ФІЛЬТРАЦІЇ РЕГІОНУ (НОВЕ) ---
st.sidebar.markdown("---")
st.sidebar.header("2. Вибір Території")

# Отримуємо список усіх доступних регіонів у файлі
available_regions = sorted(df['Region'].unique())

# Перемикач режимів
region_mode = st.sidebar.radio(
    "Як аналізувати дані?",
    ["📍 Конкретний регіон", "🇺🇦 Вся Україна (Середнє)", "✅ Обрати кілька регіонів"]
)

if region_mode == "📍 Конкретний регіон":
    # Стандартний вибір одного регіону
    selected_region = st.sidebar.selectbox("Оберіть регіон:", available_regions)
    df = df[df['Region'] == selected_region]
    st.sidebar.success(f"Дані відфільтровано: {selected_region}")

elif region_mode == "🇺🇦 Вся Україна (Середнє)":
    # Перевіряємо, чи є вже готовий рядок "Україна" в даних
    if "Україна" in available_regions:
        df = df[df['Region'] == "Україна"]
        st.sidebar.info("Використовується статистика по Україні (з файлу).")
    else:
        # Якщо немає, рахуємо середнє по всіх регіонах, що є
        st.sidebar.info("Розраховуємо середню ціну по всіх доступних регіонах...")
        # Групуємо по Даті та Продукту, рахуємо середнє ціни
        df = df.groupby(['Date', 'Product_Name'])['Price'].mean().reset_index()
        df['Region'] = 'Вся Україна (Avg)'

elif region_mode == "✅ Обрати кілька регіонів":
    # Мультиселект
    selected_regions = st.sidebar.multiselect("Оберіть регіони для об'єднання:", available_regions)
    
    if not selected_regions:
        st.error("Будь ласка, оберіть хоча б один регіон!")
        st.stop()
    else:
        # Фільтруємо вибрані, потім рахуємо середнє
        df = df[df['Region'].isin(selected_regions)]
        df = df.groupby(['Date', 'Product_Name'])['Price'].mean().reset_index()
        df['Region'] = 'Середнє по вибраним'
        st.sidebar.success(f"Об'єднано регіонів: {len(selected_regions)}")

# --- 2. EDA (АНАЛІЗ ДАНИХ) ---
st.header("Аналіз Даних")

if df is not None and not df.empty:
    # Статистика
    st.subheader("Огляд даних")
    st.write(f"Діапазон дат: з {df['Date'].min().date()} по {df['Date'].max().date()}")
    st.write(f"Всього записів: {len(df)}")
    
    # Перевірка на пропуски
    missing_count = df['Price'].isna().sum()
    if missing_count > 0:
        st.warning(f"Знайдено {missing_count} пропусків у цінах. Виправлено!")
        df['Price'] = df.groupby('Product_Name')['Price'].fillna(method='ffill')
        # Якщо на початку є пропуски, заповнюємо 'bfill'
        df['Price'] = df.groupby('Product_Name')['Price'].fillna(method='bfill')
    
    # Вибір продуктів для візуалізації
    all_products = df['Product_Name'].unique()
    selected_products_viz = st.multiselect("Оберіть продукти для порівняння графіків:", all_products, default=all_products[:2])
    
    if selected_products_viz:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df[df['Product_Name'].isin(selected_products_viz)], x='Date', y='Price', hue='Product_Name', ax=ax)
        plt.title("Динаміка цін")
        plt.grid(True)
        st.pyplot(fig)

    
    # --- 3. ПРОГНОЗ ТА МОДЕЛЮВАННЯ ---
st.header("Прогноз")

col_main1, col_main2 = st.columns([1, 3])

with col_main1:
    target_product = st.selectbox("Продукт для прогнозу:", all_products)
    
    st.markdown("---")
    st.markdown("**Налаштування Моделі**")
    
    # --- ВИБІР МОДЕЛІ ---
    model_type = st.selectbox(
        "Оберіть алгоритм прогнозу:",
        ["ARIMA (Класичний)", "Holt-Winters (Трендовий)", "SARIMA (Сезонний професійний)"]
    )
    
    # Параметри змінюються залежно від моделі
    if model_type == "ARIMA (Класичний)":
        # p - Autoregression
        p = st.number_input(
            "p (AR - Пам'ять)", 
            min_value=0, max_value=24, value=2,
            help="На скільки місяців назад дивитися? \n\n"
                 "• 1-2: Ціна залежить від останніх місяців.\n"
                 "• 12: Ціна повторює минулорічну (сезонність)."
        )
        
        # d - Integration
        d = st.number_input(
            "d (I - Тренд)", 
            min_value=0, max_value=2, value=1,
            help="Як поводиться ціна глобально?\n\n"
                 "• 0: Ціна стабільна (коливається навколо однієї суми).\n"
                 "• 1: Ціна постійно росте або падає (стандарт для інфляції).\n"
                 "• 2: Швидкий, прискорений ріст."
        )
        
        # q - Moving Average
        q = st.number_input(
            "q (MA - Згладжування)", 
            min_value=0, max_value=24, value=2,
            help="Як реагувати на раптові стрибки?\n\n"
                 "• 0: Реагувати миттєво (графік рваний).\n"
                 "• 1-3: Згладжувати випадкові коливання."
        )

        with st.expander("ℹ️ Як підібрати параметри? (Шпаргалка)"):
            st.markdown("""
            * **Для стабільних товарів:** p=1, d=0, q=1
            * **Для товарів, що дорожчають (інфляція):** p=2, d=1, q=2
            * **Для сезонних (овочі/фрукти):** Спробуйте збільшити p до 12.
            """)

    elif model_type == "SARIMA (Сезонний професійний)":
        st.info("Налаштування складаються з двох частин: Звичайні та Сезонні (Річні).")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Звичайні (p,d,q)**")
            p = st.number_input("p", 0, 5, 1, key="sp", help="Як поточний місяць залежить від попереднього.")
            d = st.number_input("d", 0, 2, 1, key="sd", help="Чи є загальний тренд росту/спаду?")
            q = st.number_input("q", 0, 5, 1, key="sq", help="Корекція помилок минулого місяця.")
        
        with col_s2:
            st.markdown("**Сезонні (P,D,Q)**")
            P = st.number_input("P (Сезонний)", 0, 5, 1, help="Зв'язок з цим же місяцем минулого року.")
            D = st.number_input("D (Сезонний)", 0, 2, 1, help="Чи змінюється сезонність з роками?")
            Q = st.number_input("Q (Сезонний)", 0, 5, 0, help="Корекція сезонних викидів.")
            s = st.number_input("s (Період)", 2, 24, 12, help="12 для місячних даних (річний цикл).")

    else:
        # Для Holt-Winters параметри простіші
        seasonal_periods = st.slider("Сезонність (міс)", 6, 24, 12, help="12 для річної сезонності")
        trend_type = st.selectbox("Тип тренду", ["add", "mul"], index=0, help="'add' для стабільного росту, 'mul' для прискореного")
    
    forecast_steps = st.slider("Період прогнозу (міс)", 1, 12, 6, help="На скільки місяців вперед робити прогноз? (таблиця)")
    
    run_btn = st.button("🔴 Розрахувати Прогноз")

with col_main2:
    if run_btn:
        # Підготовка даних
        df_prod = df[df['Product_Name'] == target_product].set_index('Date')['Price']
        # Важливо: Holt-Winters вимагає строгої частоти без пропусків
        df_prod = df_prod.asfreq('MS').fillna(method='ffill')

        try:
            # Розбиття на тест/трейн
            test_size = 6
            if len(df_prod) > test_size * 2:
                train, test = df_prod[:-test_size], df_prod[-test_size:]
            else:
                train, test = df_prod, None

            st.subheader(f"Результат ({model_type}): {target_product}")

            # --- ЛОГІКА МОДЕЛЕЙ ---
            if model_type == "ARIMA (Класичний)":
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()
                # Прогноз на тест
                if test is not None:
                    preds_test = model_fit.forecast(steps=len(test))
                # Фінальний прогноз
                final_model = ARIMA(df_prod, order=(p, d, q))
                final_fit = final_model.fit()
                future_forecast = final_fit.forecast(steps=forecast_steps)

            elif model_type == "SARIMA (Сезонний професійний)":
                # SARIMAX приймає order=(p,d,q) і seasonal_order=(P,D,Q,s)
                model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
                model_fit = model.fit(disp=False)
                if test is not None: preds_test = model_fit.forecast(steps=len(test))

                # Фінальний прогноз
                final_model = SARIMAX(df_prod, order=(p, d, q), seasonal_order=(P, D, Q, s))
                final_fit = final_model.fit(disp=False)
                future_forecast = final_fit.forecast(steps=forecast_steps)

            else: # Holt-Winters
                # 'add' - адитивний (звичайний), 'mul' - мультиплікативний (складний відсоток)
                seasonal_type = 'add' 
                
                model = ExponentialSmoothing(
                    train, 
                    trend=trend_type, 
                    seasonal=seasonal_type, 
                    seasonal_periods=seasonal_periods
                )
                model_fit = model.fit()
                
                # Прогноз на тест
                if test is not None:
                    preds_test = model_fit.forecast(steps=len(test))
                
                # Фінальний прогноз
                final_model = ExponentialSmoothing(
                    df_prod, 
                    trend=trend_type, 
                    seasonal=seasonal_type, 
                    seasonal_periods=seasonal_periods
                )
                final_fit = final_model.fit()
                future_forecast = final_fit.forecast(steps=forecast_steps)

            # --- ВІДОБРАЖЕННЯ РЕЗУЛЬТАТІВ (Спільне для обох моделей) ---
            
            # Метрики точності
            if test is not None:
                mae = mean_absolute_error(test, preds_test)
                mape = np.mean(np.abs(preds_test - test) / np.abs(test)) * 100
                
                m1, m2 = st.columns(2)
                m1.metric("MAE (Похибка в грн)", f"{mae:.2f}")
                m2.metric("MAPE (Похибка в %)", f"{mape:.2f}%")
                
                # Пояснення для користувача
                if mape < 5:
                    st.success("✅ Висока точність прогнозу!")
                elif mape < 15:
                    st.warning("⚠️ Середня точність. Можливі відхилення.")
                else:
                    st.error("❌ Низька точність. Спробуйте іншу модель або параметри.")

            # Графік
            fig_res, ax_res = plt.subplots(figsize=(10, 5))
            
            # Показуємо історію
            start_plot = df_prod.index[-36] if len(df_prod) > 36 else df_prod.index[0]
            ax_res.plot(df_prod[start_plot:].index, df_prod[start_plot:], label='Історичні дані')
            
            if test is not None:
                 ax_res.plot(test.index, preds_test, color='green', linestyle='--', label='Тест (перевірка)')
                 
            # Прогноз
            ax_res.plot(future_forecast.index, future_forecast, color='red', marker='o', linewidth=2, label=f'Прогноз ({model_type})')
            
            ax_res.legend()
            ax_res.grid(True, alpha=0.3)
            ax_res.set_title(f"Прогноз ціни на {forecast_steps} міс.")
            st.pyplot(fig_res)

            # Таблиця
            with st.expander("Переглянути точні цифри прогнозу"):
                res_df = pd.DataFrame({'Дата': future_forecast.index, 'Прогнозована ціна': future_forecast.values})
                st.dataframe(res_df.style.format({"Прогнозована ціна": "{:.2f}"}))

        except Exception as e:
            st.error(f"Помилка розрахунку: {e}. Спробуйте змінити параметри або тип тренду.")
    
st.markdown("---")