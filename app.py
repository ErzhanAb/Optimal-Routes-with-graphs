import datetime
import pickle

from catboost import CatBoostRegressor
import folium
import geopandas as gpd
import gradio as gr
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

print("🚀 Шаг 1: Импорт всех библиотек...")
print("✅ Импорт завершен.")


print("\n🌍 Шаг 2: Загрузка предобработанных данных о городе...")
with open("graph_data.pkl", "rb") as f:
    city_data = pickle.load(f)

G = city_data["graph"]
gdf_edges = city_data["edges_gdf"]
signal_nodes_set = city_data["signal_nodes"]

print(f"   ✅ Граф ({G.number_of_nodes()} узлов, {G.number_of_edges()} ребер) загружен.")
print(f"   ✅ Данные о дорогах ({len(gdf_edges)} строк) загружены.")
print(f"   ✅ Данные о светофорах ({len(signal_nodes_set)} перекрестков) загружены.")
print("✅ Данные о городе полностью готовы!")


print("\n🧠 Шаг 3: Загрузка обученной модели предсказания трафика...")
model_filename = "bishkek_traffic_model.cbm"
model = CatBoostRegressor()
model.load_model(model_filename)
print("✅ Модель успешно загружена.")


print("\n🧩 Шаг 4: Сборка функций-компонентов навигатора...")


def predict_graph_weights(graph_edges_df, model, timestamp):
    df = graph_edges_df.copy()
    df['hour'] = timestamp.hour
    df['minute'] = timestamp.minute
    df['day_of_week'] = timestamp.weekday()
    df.rename(columns={'highway': 'highway_cat'}, inplace=True)

    required_features = [
        'lanes', 'maxspeed', 'length', 'poi_count', 'highway_cat',
        'hour', 'minute', 'day_of_week'
    ]
    for col in required_features:
        if col not in df.columns:
            df[col] = 0
    predictions = model.predict(df[required_features])
    return predictions


def find_two_routes(G, start_node, end_node, weight='travel_time'):
    try:
        route1 = nx.shortest_path(
            G, source=start_node, target=end_node, weight=weight
        )
        time1 = nx.shortest_path_length(
            G, source=start_node, target=end_node, weight=weight
        )
    except nx.NetworkXNoPath:
        return None, None, None, None

    G_penalized = G.copy()
    for u, v in zip(route1[:-1], route1[1:]):
        if G_penalized.has_edge(u, v):
            G_penalized[u][v][0][weight] *= 2
    try:
        route2 = nx.shortest_path(
            G_penalized, source=start_node, target=end_node, weight=weight
        )
        time2 = sum(G[u][v][0][weight] for u, v in zip(route2[:-1], route2[1:]))
    except nx.NetworkXNoPath:
        route2, time2 = None, None
    return route1, time1, route2, time2


def plot_routes_on_folium_map(G, route1, route2, start_point, end_point):
    map_center = [
        (start_point[0] + end_point[0]) / 2,
        (start_point[1] + end_point[1]) / 2
    ]
    m = folium.Map(location=map_center, zoom_start=13, tiles="OpenStreetMap")

    if route1:
        points1 = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route1]
        folium.PolyLine(
            points1, color="white", weight=11, opacity=0.7
        ).add_to(m)
        folium.PolyLine(
            points1, color="#2ca02c", weight=6, opacity=0.85, tooltip="Маршрут 1"
        ).add_to(m)

    if route2 and route1 != route2:
        points2 = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route2]
        folium.PolyLine(
            points2, color="white", weight=11, opacity=0.7
        ).add_to(m)
        folium.PolyLine(
            points2, color="#007FFF", weight=6, opacity=0.85, tooltip="Маршрут 2"
        ).add_to(m)

    folium.Marker(
        location=start_point, popup="Точка А (Старт)", icon=folium.Icon(color="green")
    ).add_to(m)
    folium.Marker(
        location=end_point, popup="Точка Б (Финиш)", icon=folium.Icon(color="red")
    ).add_to(m)
    return m


print("✅ Вспомогательные функции готовы.")


print("\n🚀 Шаг 5: Настройка и запуск веб-интерфейса Gradio...")
geolocator = Nominatim(user_agent="bishkek_navigator_app_v2")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

SIGNAL_DELAY_SECONDS = 50


def find_and_plot_routes_by_address(
        start_address, end_address, day_of_week, hour, minute
):
    if not start_address or not end_address:
        return None, "Ошибка: Пожалуйста, введите оба адреса."
    if hour is None or minute is None:
        return None, "Ошибка: Пожалуйста, установите значения для часа и минуты."

    try:
        location_a = geocode(f"{start_address}, Бишкек")
        location_b = geocode(f"{end_address}, Бишкек")
        if not location_a or not location_b:
            return None, "Не удалось найти один или оба адреса."

        start_point = (location_a.latitude, location_a.longitude)
        end_point = (location_b.latitude, location_b.longitude)
        start_node = ox.nearest_nodes(G, Y=start_point[0], X=start_point[1])
        end_node = ox.nearest_nodes(G, Y=end_point[0], X=end_point[1])

        selected_time = datetime.datetime(2023, 1, 2 + day_of_week, hour, minute)
        travel_times = predict_graph_weights(gdf_edges, model, selected_time)
        nx.set_edge_attributes(
            G,
            values=pd.Series(travel_times, index=gdf_edges.index).to_dict(),
            name='travel_time'
        )

        r1, t1, r2, t2 = find_two_routes(G, start_node, end_node)
        if r1 is None:
            return None, "Не удалось построить маршрут."

        signals_on_route1 = sum(1 for node in r1 if node in signal_nodes_set)
        t1_total = t1 + signals_on_route1 * SIGNAL_DELAY_SECONDS
        distance1_km = sum(G[u][v][0]['length'] for u, v in zip(r1[:-1], r1[1:])) / 1000

        t2_total, distance2_km, signals_on_route2 = None, None, 0
        if r2 and r1 != r2:
            signals_on_route2 = sum(1 for node in r2 if node in signal_nodes_set)
            t2_total = t2 + signals_on_route2 * SIGNAL_DELAY_SECONDS
            distance2_km = sum(G[u][v][0]['length'] for u, v in zip(r2[:-1], r2[1:])) / 1000

        final_map = plot_routes_on_folium_map(G, r1, r2, start_point, end_point)

        t1_min_total = t1_total / 60
        output_text = (
            f"Маршрут 1:\n"
            f"  - Время: ~{t1_min_total:.1f} мин.\n"
            f"  - Расстояние: {distance1_km:.2f} км\n"
            f"  - Светофоров: {signals_on_route1}\n"
        )
        if t2_total is not None:
            t2_min_total = t2_total / 60
            output_text += (
                f"\nМаршрут 2:\n"
                f"  - Время: ~{t2_min_total:.1f} мин.\n"
                f"  - Расстояние: {distance2_km:.2f} км\n"
                f"  - Светофоров: {signals_on_route2}"
            )
        else:
            output_text += "\nАльтернативный маршрут не найден."

        return final_map._repr_html_(), output_text

    except Exception as e:
        return None, f"Произошла внутренняя ошибка: {e}"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗺️ Умный навигатор по Бишкеку")
    with gr.Row():
        with gr.Column(scale=1):
            start_address_input = gr.Textbox(
                label="Откуда?", placeholder="Например, 7 микрорайон"
            )
            end_address_input = gr.Textbox(
                label="Куда?", placeholder="Например, Ошский базар"
            )
            day_dropdown = gr.Dropdown(
                label="День недели",
                choices=[
                    "Понедельник", "Вторник", "Среда", "Четверг",
                    "Пятница", "Суббота", "Воскресенье"
                ],
                value="Понедельник",
                type="index"
            )
            hour_slider = gr.Slider(
                label="Час", minimum=0, maximum=23, step=1, value=8
            )
            minute_slider = gr.Slider(
                label="Минута", minimum=0, maximum=59, step=1, value=30
            )
            build_btn = gr.Button("🚀 Построить маршруты", variant="primary")
            output_textbox = gr.Textbox(
                label="Информация о маршрутах", interactive=False, lines=7
            )
        with gr.Column(scale=2):
            output_map_html = gr.HTML(label="Карта с маршрутами")

    build_btn.click(
        fn=find_and_plot_routes_by_address,
        inputs=[
            start_address_input, end_address_input, day_dropdown,
            hour_slider, minute_slider
        ],
        outputs=[output_map_html, output_textbox]
    )

demo.launch()
print("\n\n🌐 Веб-интерфейс запущен!")
