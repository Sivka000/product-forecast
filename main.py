import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    # --- 3. НАВЧАННЯ МОДЕЛІ (ARIMA) ---
    st.header("Прогноз")
    
    target_product = st.selectbox("Оберіть продукт для прогнозування:", all_products)
    
    # Підготовка ряду
    df_prod = df[df['Product_Name'] == target_product].set_index('Date')['Price']
    df_prod = df_prod.asfreq('MS') # Встановлюємо частоту (початок місяця)
    
    # Якщо після asfreq з'явилися NaN (через пропущені місяці), заповнюємо їх
    if df_prod.isna().sum() > 0:
         df_prod = df_prod.fillna(method='ffill')

    # Налаштування параметрів
    st.sidebar.subheader("Параметри моделі ARIMA")
    p = st.sidebar.number_input("p (Autoregression)", 0, 10, 5, key='p')
    d = st.sidebar.number_input("d (Integration)", 0, 5, 3, key='d')
    q = st.sidebar.number_input("q (Moving Average)", 0, 10, 5, key='q')
    forecast_steps = st.sidebar.slider("Період прогнозу (міс)", 1, 12, 12)

    if st.button("Розрахувати прогноз"):
        with st.spinner('Тренування моделі...'):
            try:
                # Розділення на тренувальну і тестову (останні 6 місяців для тесту)
                test_size = 6
                if len(df_prod) > test_size * 2:
                    train = df_prod[:-test_size]
                    test = df_prod[-test_size:]
                else:
                    train = df_prod
                    test = None

                # Навчання моделі
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()

                # Оцінка
                col1, col2 = st.columns(2)
                
                if test is not None:
                    predictions_test = model_fit.forecast(steps=len(test))
                    mae = mean_absolute_error(test, predictions_test)
                    mape = np.mean(np.abs(predictions_test - test) / np.abs(test)) * 100
                    
                    with col1:
                        st.subheader("Точність (на тестових даних)")
                        st.metric("Середня похибка (MAE)", f"{mae:.2f} грн")
                        st.metric("Відсоток похибки (MAPE)", f"{mape:.2f}%")
                
                # Фінальний прогноз на майбутнє
                final_model = ARIMA(df_prod, order=(p, d, q))
                final_fit = final_model.fit()
                future_forecast = final_fit.forecast(steps=forecast_steps)
                
                # Вивід таблиці прогнозу
                future_df = pd.DataFrame({
                    'Дата': future_forecast.index,
                    'Прогноз ціни': future_forecast.values
                })
                
                with col2:
                    st.subheader(f"Прогноз на {forecast_steps} міс.")
                    st.dataframe(future_df.style.format({"Прогноз ціни": "{:.2f}"}))

                # Графік
                st.subheader("Візуалізація Прогнозу")
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                
                # Історія (останні 2 роки для кращої видимості)
                display_start_date = df_prod.index[-24] if len(df_prod) > 24 else df_prod.index[0]
                history_subset = df_prod[df_prod.index >= display_start_date]
                
                ax2.plot(history_subset.index, history_subset, label='Історичні дані')
                
                if test is not None:
                    # Показуємо, як модель вгадала тестовий період
                    ax2.plot(test.index, predictions_test, color='green', linestyle='--', label='Тестовий прогноз (перевірка)')
                
                # Майбутній прогноз
                ax2.plot(future_forecast.index, future_forecast, color='red', marker='o', label='Прогноз на майбутнє')
                
                ax2.set_title(f"Прогноз ціни: {target_product}")
                ax2.set_ylabel("Ціна (грн)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Помилка при навчанні моделі: {e}. Спробуйте змінити параметри p, d, q.")

st.markdown("---")