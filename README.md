# 🗺️ Bishkek Smart Navigator

This is a predictive navigator project for the city of Bishkek that calculates optimal routes based on predicted travel times. The travel time for each road segment is predicted using a CatBoost machine learning model trained on simulated traffic data.

[![Hugging Face Spaces](https://huggingface.co/spaces/ErzhanAb/Optimal-Routes-with-graphs)](https://huggingface.co/spaces/ErzhanAb/Optimal-Routes-with-graphs)

## 🚀 Key Features

-   **Machine Learning for Traffic Prediction**: Utilizes a `CatBoostRegressor` model to estimate travel time based on the day of the week, time of day, road type, number of lanes, and density of Points of Interest (POI).
-   **Geo-Data Enrichment**: The road graph from OpenStreetMap is enriched with data on traffic light locations and POIs (cafes, offices, schools) for more accurate modeling.
-   **Alternative Route Generation**: The application finds not only the fastest route but also a viable alternative.
-   **Interactive Web Interface**: The user interface is built with Gradio, allowing for easy input of addresses and selection of travel times.
-   **Optimized for Fast Startup**: All resource-intensive operations for loading and processing the city map are pre-computed and saved to a single file, ensuring a near-instant application launch.

## ⚙️ How It Works: Project Architecture

The project consists of two main stages: data pre-processing (performed once) and the web application's operation (runs on every launch).

### 1. Offline Pre-processing Stage

This stage was completed in a Google Colab environment to prepare the data that the application would use.

1.  **Data Collection**: Using the `OSMnx` library, the following were downloaded from OpenStreetMap:
    -   The road graph for Bishkek (`network_type='drive'`).
    -   Coordinates of all traffic lights (`highway='traffic_signals'`).
    -   Coordinates of Points of Interest (POIs) such as cafes, shops, banks, etc.
2.  **Feature Engineering**:
    -   Basic road attributes (speed limit, number of lanes) were cleaned, and missing values were filled.
    -   A new feature was calculated for each road segment: **POI density** (the number of points of interest within a 50-meter radius).
    -   Traffic lights were mapped to the nearest nodes in the graph.
3.  **Traffic Data Simulation**: Since real historical traffic data was unavailable, a simulator function was written to generate realistic travel times based on multiple factors:
    -   **Time of Day**: Morning and evening rush hours.
    -   **Day of the Week**: Differences between weekdays and weekends.
    -   **Road Characteristics**: Type (primary, secondary), speed limit, length.
    -   **POI Density**: Streets with more establishments are considered more congested.
4.  **Model Training**:
    -   A `CatBoostRegressor` model was trained on the generated dataset.
    -   Target variable: `travel_time`.
    -   Features: `hour`, `day_of_week`, `length`, `maxspeed`, `lanes`, `poi_count`, `highway_type`.
5.  **Saving Artifacts**:
    -   The trained model was saved to the file `bishkek_traffic_model.cbm`.
    -   The processed graph, road data, and traffic light information were saved into a single `graph_data.pkl` file using `pickle`.

### 2. Online Web Application

The main `app.py` file, which is deployed on Hugging Face Spaces.

1.  **Startup**: The application instantly loads the pre-processed files `graph_data.pkl` and `bishkek_traffic_model.cbm`.
2.  **User Input**: The user enters departure/destination addresses and selects a travel time via the Gradio interface.
3.  **Geocoding**: Addresses are converted into geographic coordinates (latitude and longitude) using `geopy`.
4.  **Real-time Prediction**:
    -   Based on the user-selected time, the model predicts the `travel_time` for **every** road segment in the city.
    -   These predicted values are set as the "weights" for the edges in the road graph.
5.  **Route Finding**: Using the `networkx` library, the shortest path (Dijkstra's algorithm) is found on the weighted graph from the start node to the end node.
6.  **Visualization**: The found routes are drawn on an interactive `folium` map, which is displayed to the user along with information about travel time and distance.

## 📁 Repository Structure

*   🐍 `app.py` - The main Gradio web application code.
*   📊 `bishkek_traffic_model.cbm` - The trained CatBoost model.
*   🏙️ `graph_data.pkl` - Pre-processed city graph data (roads, POIs, traffic lights).
*   M `README.md` - This project description file.
*   📋 `requirements.txt` - A list of Python dependencies to install.

## 🛠️ Setup and Deployment

### Option 1: Local Setup (On Your Computer)

This method allows you to run the application on your own machine without needing to register for any services.

1.  **Download the Project Files**
    -   Click the green `<> Code` button at the top of this repository.
    -   Select `Download ZIP` from the dropdown menu.

2.  **Unzip the Archive**
    -   Extract the downloaded ZIP archive to a convenient folder.

3.  **Install Dependencies**
    -   Ensure you have Python installed (version 3.8 or newer).
    -   Open a terminal (command prompt) and navigate to the project folder.
    -   It is recommended to create a virtual environment:
        ```bash
        # Create the environment
        python -m venv venv
        # Activate it (for Windows)
        venv\Scripts\activate
        # Activate it (for MacOS/Linux)
        source venv/bin/activate
        ```
    -   Install all required libraries with a single command:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Run the Application**
    -   In the same terminal, execute the command:
        ```bash
        python app.py
        ```

5.  **Done!**
    -   A link will appear in the terminal, usually `http://127.0.0.1:7860`. Open it in your web browser to see the navigator in action.

### Option 2: Cloud Deployment on Hugging Face Spaces

This method allows you to publish your application on the internet.

1.  **Create a new Space**
    -   Register or log in to your account on [Hugging Face](https://huggingface.co/).
    -   Click **New Space**.
    -   **SDK**: Select **Gradio**.
    -   **Hardware**: Leave it as `CPU basic` (free).
    -   Click **Create Space**.

2.  **Upload the Files**
    -   On your new Space's page, go to the `Files` tab.
    -   Click `Add file` -> `Upload files`.
    -   Drag and drop all the project files from your computer into the upload window: `app.py`, `requirements.txt`, `graph_data.pkl`, and `bishkek_traffic_model.cbm`.
    -   Click **Commit changes to main**.

3.  **Done!**
    -   Hugging Face will automatically start building the application (`Building`), which may take 5-10 minutes.
    -   Once the status changes to `Running`, your application will be available at a public link.

## 🔮 Potential Improvements

-   **Using real traffic data** instead of simulated data (e.g., via APIs from services like Yandex.Maps or 2GIS, if available).
-   **Adding new features** to the model, such as weather conditions, public holidays, and road work.
-   **Caching** geocoded addresses to speed up repeated requests.
-   **Expanding to other cities**: creating a universal pre-processing script that can prepare data for any city.

---
