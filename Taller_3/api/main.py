from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: str
    sex: str


def load_artifacts():
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", "/opt/artifacts")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    model_path = os.path.join(artifacts_dir, "model.joblib")
    label_encoder_path = os.path.join(artifacts_dir, "label_encoder.joblib")
    training_data_path = os.path.join(artifacts_dir, "training_data_for_viz.joblib")
    pca_path = os.path.join(artifacts_dir, "pca.joblib")

    if not (
        os.path.exists(preprocessor_path)
        and os.path.exists(model_path)
        and os.path.exists(label_encoder_path)
        and os.path.exists(training_data_path)
        and os.path.exists(pca_path)
    ):
        raise FileNotFoundError(
            "Artifacts not found. Ensure the Airflow DAG has trained and saved all artifacts."
        )

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    training_data_for_viz = joblib.load(training_data_path)
    pca = joblib.load(pca_path)

    return preprocessor, model, label_encoder, training_data_for_viz, pca


app = FastAPI(title="Penguins Inference API")

# --- MODIFIED: Lazy loading model artifacts ---
PREPROCESSOR, MODEL, LABEL_ENCODER, TRAINING_DATA_VIZ, PCA_MODEL = None, None, None, None, None
ARTIFACTS_LOADED_SUCCESSFULLY = False

def load_artifacts_if_needed():
    """
    Load artifacts into global vars. If successful, it won't try again.
    If it fails, it will retry on the next call.
    """
    global PREPROCESSOR, MODEL, LABEL_ENCODER, TRAINING_DATA_VIZ, PCA_MODEL, ARTIFACTS_LOADED_SUCCESSFULLY
    if ARTIFACTS_LOADED_SUCCESSFULLY:
        return

    try:
        PREPROCESSOR, MODEL, LABEL_ENCODER, TRAINING_DATA_VIZ, PCA_MODEL = load_artifacts()
        ARTIFACTS_LOADED_SUCCESSFULLY = True
    except FileNotFoundError as e:
        # Re-raise as HTTPException to be handled by FastAPI
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Please run the training pipeline first.",
        ) from e

@app.post("/predict")
def predict(features: PenguinFeatures):
    try:
        load_artifacts_if_needed()
    except HTTPException as e:
        # Propagate the 503 error if artifacts are not ready
        raise e

    if any(value is None for value in features.dict().values()):
        raise HTTPException(status_code=400, detail="All fields must be provided.")

    df = pd.DataFrame([features.dict()])

    try:
        # --- PREDICTION ---
        X_processed = PREPROCESSOR.transform(df)
        y_pred_encoded = MODEL.predict(X_processed)
        species = LABEL_ENCODER.inverse_transform(y_pred_encoded)[0]

        # --- NEIGHBOR & PCA ANALYSIS ---
        X_train_orig = TRAINING_DATA_VIZ["features_raw"]
        y_train_encoded = TRAINING_DATA_VIZ["labels_encoded"]
        X_train_pca = TRAINING_DATA_VIZ["features_pca"]
        
        X_train_processed = PREPROCESSOR.transform(X_train_orig)

        nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        nn.fit(X_train_processed)
        distances, indices = nn.kneighbors(X_processed)

        neighbor_indices = indices[0]
        neighbor_features = X_train_orig.iloc[neighbor_indices].to_dict(orient="records")
        neighbor_labels_encoded = y_train_encoded[neighbor_indices]
        neighbor_species = LABEL_ENCODER.inverse_transform(neighbor_labels_encoded)

        neighbors = []
        for i in range(len(neighbor_indices)):
            neighbors.append(
                {"features": neighbor_features[i], "species": neighbor_species[i]}
            )
        
        confidence = np.sum(neighbor_species == species) / len(neighbor_species)

        # Transform the new point and the training data to 2D for plotting
        new_point_pca = PCA_MODEL.transform(X_processed)
        
        # Prepare data for chart
        viz_data = {
            "new_point": {"x": new_point_pca[0, 0], "y": new_point_pca[0, 1]},
            "training_points": X_train_pca.to_dict(orient="records"),
            "training_labels": LABEL_ENCODER.inverse_transform(y_train_encoded).tolist(),
            "neighbor_indices": neighbor_indices.tolist(),
        }

    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "predicted_species": species,
        "confidence": confidence,
        "neighbors": neighbors,
        "viz_data": viz_data,
    }


@app.get("/warmup")
def warmup():
    try:
        load_artifacts_if_needed()
        return {"status": "ok", "artifacts_loaded": True}
    except HTTPException as e:
        # translate to consistent shape
        return {"status": "unavailable", "detail": e.detail}


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Penguins Classifier</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial; margin: 24px; display: grid; grid-template-columns: 520px 1fr; gap: 32px; }
      form { max-width: 520px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      h1 { margin-bottom: 16px; grid-column: 1 / -1; }
      label { font-weight: 600; }
      input, select { padding: 8px; }
      .row { display: contents; }
      .full { grid-column: 1 / -1; }
      button { padding: 10px 14px; font-weight: 600; cursor: pointer; }
      #result-container { grid-column: 1 / -1; }
      #result { margin-top: 16px; font-size: 1.2rem; font-weight: 700; opacity: 0; transition: opacity 0.5s; }
      #explanation { margin-top: 24px; opacity: 0; transition: opacity 0.5s; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
      th { background-color: #f2f2f2; }
      #pca-chart-container { width: 100%; max-width: 800px; }
    </style>
  </head>
  <body>
    <main>
      <form id="form">
        <h1>Penguins Classifier</h1>
        <div class="row">
          <label>Bill length (mm)</label>
          <input type="number" step="0.1" name="bill_length_mm" required />
        </div>
        <div class="row">
          <label>Bill depth (mm)</label>
          <input type="number" step="0.1" name="bill_depth_mm" required />
        </div>
        <div class="row">
          <label>Flipper length (mm)</label>
          <input type="number" step="1" name="flipper_length_mm" required />
        </div>
        <div class="row">
          <label>Body mass (g)</label>
          <input type="number" step="1" name="body_mass_g" required />
        </div>
        <div class="row">
          <label>Island</label>
          <select name="island" required>
            <option value="Torgersen">Torgersen</option>
            <option value="Biscoe">Biscoe</option>
            <option value="Dream">Dream</option>
          </select>
        </div>
        <div class="row">
          <label>Sex</label>
          <select name="sex" required>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
        </div>
        <div class="full" style="display:flex; gap:8px;">
          <button type="submit">Classify</button>
          <button type="button" id="random">Random</button>
        </div>
        <div id="result-container">
          <div id="result"></div>
          <div id="explanation"></div>
        </div>
      </form>
    </main>
    <aside>
      <div id="pca-chart-container">
        <canvas id="pcaChart"></canvas>
      </div>
    </aside>
    <script>
      const form = document.getElementById('form');
      const resultDiv = document.getElementById('result');
      const explanationDiv = document.getElementById('explanation');
      const ctx = document.getElementById('pcaChart').getContext('2d');
      let pcaChart;

      function randomBetween(min, max, step=1) { const n = Math.floor((Math.random()*(max-min))/step)*step + min; return Number(n.toFixed(2)); }
      document.getElementById('random').addEventListener('click', () => {
        form.bill_length_mm.value = randomBetween(32, 60, 0.1);
        form.bill_depth_mm.value = randomBetween(13, 22, 0.1);
        form.flipper_length_mm.value = randomBetween(170, 235, 1);
        form.body_mass_g.value = randomBetween(2700, 6300, 10);
        const islands = ['Torgersen','Biscoe','Dream'];
        const sexes = ['Male','Female'];
        form.island.value = islands[Math.floor(Math.random()*islands.length)];
        form.sex.value = sexes[Math.floor(Math.random()*sexes.length)];
      });

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());
        for (const key of Object.keys(payload)) {
          if (["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"].includes(key)) {
            payload[key] = Number(payload[key]);
          }
        }
        resultDiv.textContent = 'Predicting...';
        resultDiv.style.opacity = 1;
        explanationDiv.innerHTML = '';
        explanationDiv.style.opacity = 0;

        try {
          const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
          const data = await res.json();
          
          if (!res.ok) {
            resultDiv.textContent = `Error: ${data.detail || 'Request failed'}`;
            if (pcaChart) { pcaChart.destroy(); }
            return;
          }

          resultDiv.textContent = `Predicted species: ${data.predicted_species}`;

          let explanationHTML = `
            <h3>Confidence: ${Math.round(data.confidence * 100)}%</h3>
            <p>Based on the ${data.neighbors.length} most similar penguins from the training data.</p>
            <h4>Nearest Neighbors:</h4>
            <table>
              <thead>
                <tr>
                  <th>Species</th><th>Bill Length</th><th>Bill Depth</th><th>Flipper Length</th><th>Body Mass</th><th>Island</th><th>Sex</th>
                </tr>
              </thead>
              <tbody>
          `;
          data.neighbors.forEach(n => {
            explanationHTML += `
              <tr>
                <td>${n.species}</td><td>${n.features.bill_length_mm}</td><td>${n.features.bill_depth_mm}</td><td>${n.features.flipper_length_mm}</td><td>${n.features.body_mass_g}</td><td>${n.features.island}</td><td>${n.features.sex}</td>
              </tr>
            `;
          });
          explanationHTML += '</tbody></table>';
          explanationDiv.innerHTML = explanationHTML;
          explanationDiv.style.opacity = 1;

          renderChart(data.viz_data, data.predicted_species);

        } catch (err) {
          resultDiv.textContent = 'Prediction failed. Check browser console for details.';
          if (pcaChart) { pcaChart.destroy(); }
        }
      });

      function renderChart(vizData, predictedSpecies) {
        if (pcaChart) {
          pcaChart.destroy();
        }

        const speciesColors = {
          'Adelie': 'rgba(255, 99, 132, 0.6)',
          'Chinstrap': 'rgba(54, 162, 235, 0.6)',
          'Gentoo': 'rgba(75, 192, 192, 0.6)',
        };
        const pointStyles = {
          'Adelie': 'circle',
          'Chinstrap': 'rect',
          'Gentoo': 'triangle',
        };
        const newPointColor = 'rgba(255, 206, 86, 1)';
        const neighborColor = 'rgba(153, 102, 255, 1)';

        const datasets = [{
          label: 'New Penguin',
          data: [vizData.new_point],
          backgroundColor: newPointColor,
          pointStyle: 'star',
          radius: 15,
          borderColor: 'black',
          borderWidth: 2,
        }];

        const speciesData = {};
        vizData.training_labels.forEach((label, i) => {
          if (!speciesData[label]) {
            speciesData[label] = {
              label: label,
              data: [],
              backgroundColor: speciesColors[label],
              pointStyle: pointStyles[label],
              radius: 5,
            };
          }
          let point = {
            x: vizData.training_points[i].pca1,
            y: vizData.training_points[i].pca2,
            isNeighbor: vizData.neighbor_indices.includes(i)
          };
          speciesData[label].data.push(point);
        });

        Object.values(speciesData).forEach(dataset => {
            dataset.radius = dataset.data.map(p => p.isNeighbor ? 8 : 5);
            dataset.borderColor = dataset.data.map(p => p.isNeighbor ? 'black' : 'transparent');
            dataset.borderWidth = dataset.data.map(p => p.isNeighbor ? 2 : 0);
            datasets.push(dataset);
        });

        pcaChart = new Chart(ctx, {
          type: 'scatter',
          data: { datasets },
          options: {
            responsive: true,
            plugins: {
              legend: {
                position: 'top',
              },
              title: {
                display: true,
                text: 'Penguin Similarity (PCA)'
              },
              tooltip: {
                  callbacks: {
                      label: function(context) {
                          let label = context.dataset.label || '';
                          if(context.dataset.label === 'New Penguin') {
                              return `Your Penguin (${predictedSpecies})`;
                          }
                          if (context.raw.isNeighbor) {
                              label += ' (Neighbor)';
                          }
                          return label;
                      }
                  }
              }
            },
            scales: {
              x: {
                title: {
                  display: true,
                  text: 'Principal Component 1'
                }
              },
              y: {
                title: {
                  display: true,
                  text: 'Principal Component 2'
                }
              }
            }
          }
        });
      }
    </script>
  </body>
</html>
"""


