import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "ncf_params.npz"

DATA_CANDIDATES = [
    ROOT / "data" / "movielens_cleaned.csv",
    ROOT / "data" / "movielens_100k.csv",
]

    
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại:\n{model_path}"
        )

    raw = np.load(model_path, allow_pickle=False)

    params = {
        "P_gmf": raw["P_gmf"],
        "Q_gmf": raw["Q_gmf"],
        "P_mlp": raw["P_mlp"],
        "Q_mlp": raw["Q_mlp"],
        "W1": raw["W1"],
        "b1": raw["b1"],
        "W2": raw["W2"],
        "b2": raw["b2"],
        "Wout": raw["Wout"],
        "bout": raw["bout"],
    }

    user_ids = raw["user_ids"].astype(int)
    movie_ids = raw["movie_ids"].astype(int)

    global_mean = float(raw["global_mean"])

    return params, user_ids, movie_ids, global_mean


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data_path = None

    for path in DATA_CANDIDATES:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(
            "Không tìm thấy dataset.\n\n"
            "Đã kiểm tra:\n"
            + "\n".join(str(p) for p in DATA_CANDIDATES)
        )

    df = pd.read_csv(
        data_path,
        encoding="utf-8-sig"
    )

    return df


# ============================================================
# NCF PREDICTION
# ============================================================

def relu(x):
    return np.maximum(0, x)


def ncf_predict(
    params,
    user_idx_arr,
    movie_idx_arr,
    global_mean
):

    # GMF
    p_g = params["P_gmf"][user_idx_arr]
    q_g = params["Q_gmf"][movie_idx_arr]

    gmf_vec = p_g * q_g

    # MLP
    p_m = params["P_mlp"][user_idx_arr]
    q_m = params["Q_mlp"][movie_idx_arr]

    mlp_input = np.concatenate(
        [p_m, q_m],
        axis=1
    )

    h1 = relu(
        mlp_input @ params["W1"]
        + params["b1"]
    )

    h2 = relu(
        h1 @ params["W2"]
        + params["b2"]
    )

    # GMF + MLP
    final_concat = np.concatenate(
        [gmf_vec, h2],
        axis=1
    )

    prediction = (
        final_concat @ params["Wout"]
        + params["bout"]
    ).flatten()

    return prediction + global_mean


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(
    user_id,
    top_n,
    params,
    user_ids,
    movie_ids,
    ratings,
    global_mean
):

    user_id_to_idx = {
        int(uid): idx
        for idx, uid in enumerate(user_ids)
    }

    movie_id_to_idx = {
        int(mid): idx
        for idx, mid in enumerate(movie_ids)
    }

    if user_id not in user_id_to_idx:
        return pd.DataFrame()

    user_idx = user_id_to_idx[user_id]

    # Movies already rated by user
    rated_movies = set(
        ratings[
            ratings["user_id"] == user_id
        ]["movie_id"]
        .astype(int)
    )

    # Candidate movies
    candidate_movies = [
        int(movie_id)
        for movie_id in movie_ids
        if int(movie_id) not in rated_movies
    ]

    if len(candidate_movies) == 0:
        return pd.DataFrame()

    candidate_indices = np.array(
        [
            movie_id_to_idx[movie_id]
            for movie_id in candidate_movies
        ],
        dtype=np.int64
    )

    user_indices = np.full(
        len(candidate_indices),
        user_idx,
        dtype=np.int64
    )

    predictions = ncf_predict(
        params,
        user_indices,
        candidate_indices,
        global_mean
    )

    predictions = np.clip(
        predictions,
        1,
        5
    )

    recommendations = pd.DataFrame({
        "movie_id": candidate_movies,
        "predicted_rating": predictions
    })

    recommendations = (
        recommendations
        .sort_values(
            "predicted_rating",
            ascending=False
        )
        .head(top_n)
    )

    return recommendations


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    params, user_ids, movie_ids, global_mean = load_model(
        MODEL_PATH
    )

    df = load_data()

except Exception as e:

    st.error(str(e))

    st.stop()


# ============================================================
# PREPARE RATINGS
# ============================================================

ratings = df[
    ["user_id", "movie_id"]
].copy()

ratings["user_id"] = ratings["user_id"].astype(int)
ratings["movie_id"] = ratings["movie_id"].astype(int)


# ============================================================
# UI
# ============================================================

st.title("🎬 Movie Recommendation System")

st.write(
    "Interactive Movie Recommendation "
    "using Neural Collaborative Filtering"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Recommendation Settings")

    selected_user = st.selectbox(
        "User ID",
        user_ids.tolist()
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )

    st.divider()

    st.subheader("Model")

    st.write(
        "**Neural Collaborative Filtering**"
    )

    st.write(
        "GMF + MLP"
    )

    st.write(
        "GMF Embedding: 8"
    )

    st.write(
        "MLP Embedding: 8"
    )

    st.write(
        "Hidden Layers: 16 → 8"
    )

    st.divider()

    st.subheader("Dataset")

    st.write(
        f"Users: **{len(user_ids):,}**"
    )

    st.write(
        f"Movies: **{len(movie_ids):,}**"
    )

    st.write(
        f"Global Mean Rating: **{global_mean:.3f}**"
    )


# ============================================================
# MAIN
# ============================================================

st.subheader(
    f"Recommended Movies for User {selected_user}"
)


if st.button(
    "🎯 Generate Recommendations",
    type="primary"
):

    with st.spinner(
        "Generating recommendations..."
    ):

        recommendations = recommend_movies(
            selected_user,
            top_n,
            params,
            user_ids,
            movie_ids,
            ratings,
            global_mean
        )

    if recommendations.empty:

        st.warning(
            "Không tìm thấy recommendation."
        )

    else:

        st.success(
            f"Generated {len(recommendations)} "
            f"recommendations."
        )

        for rank, (_, row) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            movie_id = int(
                row["movie_id"]
            )

            rating = float(
                row["predicted_rating"]
            )

            with st.container(
                border=True
            ):

                col1, col2 = st.columns(
                    [1, 8]
                )

                with col1:

                    st.markdown(
                        f"## #{rank}"
                    )

                with col2:

                    st.markdown(
                        f"### Movie ID: {movie_id}"
                    )

                    st.write(
                        f"⭐ Predicted Rating: "
                        f"**{rating:.2f} / 5.00**"
                    )


# ============================================================
# ABOUT
# ============================================================

with st.expander(
    "ℹ️ About this Project"
):

    st.markdown(
        """
        ### Neural Collaborative Filtering

        This application uses a custom NumPy implementation
        of Neural Collaborative Filtering.

        **Architecture**

        - Generalized Matrix Factorization (GMF)
        - Multi-Layer Perceptron (MLP)
        - GMF + MLP fusion
        - Adam optimization
        - Early stopping

        **Workflow**

        User ID → NCF Model → Prediction → Top-N Recommendation
        """
    )