"""
CHƯƠNG 3.3: ĐÁNH GIÁ ĐỘ CHÍNH XÁC CỦA MÔ HÌNH
3.3.2. So sánh hiệu năng của mô hình NCF đề xuất với các mô hình cơ sở

So sánh 3 mô hình:
  1. Item-based kNN Collaborative Filtering (baseline truyền thống)
  2. Matrix Factorization / SVD (baseline model-based CF)
  3. Neural Collaborative Filtering - NCF (mô hình đề xuất, GMF + MLP)
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# 0. CHUẨN BỊ DỮ LIỆU
# ============================================================
df = pd.read_csv('/home/claude/movielens_cleaned.csv')

n_users = df['user_idx'].nunique()
n_movies = df['movie_idx'].nunique()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
print(f"Số user: {n_users} | Số phim: {n_movies}")

global_mean = train_df['rating'].mean()

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

results = []

# ============================================================
# MODEL 1: ITEM-BASED kNN COLLABORATIVE FILTERING
# ============================================================
print("\n" + "="*60)
print("MODEL 1: Item-based kNN Collaborative Filtering")
print("="*60)
t0 = time.time()

# Xây ma trận User-Item từ tập train (943 x 1681)
train_matrix = np.zeros((n_users, n_movies))
for u, i, r in zip(train_df['user_idx'], train_df['movie_idx'], train_df['rating']):
    train_matrix[u, i] = r

# Ma trận tương đồng Cosine giữa các item (dựa trên vector rating của item)
item_sim = cosine_similarity(train_matrix.T)  # (n_movies x n_movies)
np.fill_diagonal(item_sim, 0)

K_NEIGHBORS = 20

def predict_knn(u, i):
    user_rated = np.nonzero(train_matrix[u])[0]
    if len(user_rated) == 0:
        return global_mean
    sims = item_sim[i, user_rated]
    top_k_idx = np.argsort(sims)[-K_NEIGHBORS:]
    top_sims = sims[top_k_idx]
    top_ratings = train_matrix[u, user_rated[top_k_idx]]
    if top_sims.sum() <= 1e-8:
        return train_df.loc[train_df['movie_idx'] == i, 'rating'].mean() if (train_df['movie_idx'] == i).any() else global_mean
    return np.dot(top_sims, top_ratings) / top_sims.sum()

test_u = test_df['user_idx'].values
test_i = test_df['movie_idx'].values
test_r = test_df['rating'].values

knn_preds = np.array([predict_knn(u, i) for u, i in zip(test_u, test_i)])
knn_preds = np.clip(knn_preds, 1, 5)

knn_rmse = rmse(test_r, knn_preds)
knn_mae = mae(test_r, knn_preds)
knn_time = time.time() - t0
print(f"RMSE: {knn_rmse:.4f} | MAE: {knn_mae:.4f} | Thời gian: {knn_time:.1f}s")
results.append(('Item-based kNN', knn_rmse, knn_mae, knn_time))

# ============================================================
# MODEL 2: MATRIX FACTORIZATION (SVD-style / FunkSVD)
# ============================================================
print("\n" + "="*60)
print("MODEL 2: Matrix Factorization (SVD-style, Biased MF)")
print("="*60)
t0 = time.time()

K = 20          # số chiều latent factor
N_EPOCHS = 25
LR = 0.01
REG = 0.05

P = np.random.normal(0, 0.1, (n_users, K))    # user latent factors
Q = np.random.normal(0, 0.1, (n_movies, K))   # item latent factors
b_u = np.zeros(n_users)
b_i = np.zeros(n_movies)

train_u = train_df['user_idx'].values
train_i = train_df['movie_idx'].values
train_r = train_df['rating'].values

for epoch in range(N_EPOCHS):
    perm = np.random.permutation(len(train_u))
    for idx in perm:
        u, i, r = train_u[idx], train_i[idx], train_r[idx]
        pred = global_mean + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
        err = r - pred
        b_u[u] += LR * (err - REG * b_u[u])
        b_i[i] += LR * (err - REG * b_i[i])
        P_u_old = P[u].copy()
        P[u] += LR * (err * Q[i] - REG * P[u])
        Q[i] += LR * (err * P_u_old - REG * Q[i])
    if (epoch + 1) % 5 == 0:
        train_pred = global_mean + b_u[train_u] + b_i[train_i] + np.sum(P[train_u] * Q[train_i], axis=1)
        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS} - Train RMSE: {rmse(train_r, train_pred):.4f}")

mf_preds = global_mean + b_u[test_u] + b_i[test_i] + np.sum(P[test_u] * Q[test_i], axis=1)
mf_preds = np.clip(mf_preds, 1, 5)

mf_rmse = rmse(test_r, mf_preds)
mf_mae = mae(test_r, mf_preds)
mf_time = time.time() - t0
print(f"RMSE: {mf_rmse:.4f} | MAE: {mf_mae:.4f} | Thời gian: {mf_time:.1f}s")
results.append(('Matrix Factorization (SVD)', mf_rmse, mf_mae, mf_time))

# ============================================================
# MODEL 3: NEURAL COLLABORATIVE FILTERING (NCF = GMF + MLP)
# ============================================================
print("\n" + "="*60)
print("MODEL 3: Neural Collaborative Filtering (NCF - GMF + MLP)")
print("="*60)
t0 = time.time()

K_GMF = 8
K_MLP = 8
H1, H2 = 16, 8
N_EPOCHS_NCF = 40
BATCH_SIZE = 256
LR_NCF = 0.001
REG_NCF = 1e-4          # L2 regularization (chống overfitting)
PATIENCE = 5            # early stopping

# Tách thêm tập validation từ train để theo dõi overfitting
train_u_full, train_i_full, train_r_full = train_u, train_i, train_r
val_split = int(0.9 * len(train_u_full))
val_perm = np.random.permutation(len(train_u_full))
tr_idx, va_idx = val_perm[:val_split], val_perm[val_split:]
train_u, train_i, train_r = train_u_full[tr_idx], train_i_full[tr_idx], train_r_full[tr_idx]
val_u, val_i, val_r = train_u_full[va_idx], train_i_full[va_idx], train_r_full[va_idx]

def init_w(shape):
    return np.random.normal(0, np.sqrt(2.0 / shape[0]), shape)

params = {
    'P_gmf': np.random.normal(0, 0.1, (n_users, K_GMF)),
    'Q_gmf': np.random.normal(0, 0.1, (n_movies, K_GMF)),
    'P_mlp': np.random.normal(0, 0.1, (n_users, K_MLP)),
    'Q_mlp': np.random.normal(0, 0.1, (n_movies, K_MLP)),
    'W1': init_w((2 * K_MLP, H1)), 'b1': np.zeros(H1),
    'W2': init_w((H1, H2)), 'b2': np.zeros(H2),
    'Wout': init_w((K_GMF + H2, 1)), 'bout': np.zeros(1),
}
# Adam optimizer state
m_state = {k: np.zeros_like(v) for k, v in params.items()}
v_state = {k: np.zeros_like(v) for k, v in params.items()}
beta1, beta2, eps = 0.9, 0.999, 1e-8
t_step = 0

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

def adam_update(key, grad):
    global t_step
    m_state[key] = beta1 * m_state[key] + (1 - beta1) * grad
    v_state[key] = beta2 * v_state[key] + (1 - beta2) * (grad ** 2)
    m_hat = m_state[key] / (1 - beta1 ** t_step)
    v_hat = v_state[key] / (1 - beta2 ** t_step)
    params[key] -= LR_NCF * m_hat / (np.sqrt(v_hat) + eps)

def ncf_predict(u_arr, i_arr):
    p_g, q_g = params['P_gmf'][u_arr], params['Q_gmf'][i_arr]
    gmf_vec = p_g * q_g
    p_m, q_m = params['P_mlp'][u_arr], params['Q_mlp'][i_arr]
    mlp_in = np.concatenate([p_m, q_m], axis=1)
    h1 = relu(mlp_in @ params['W1'] + params['b1'])
    h2 = relu(h1 @ params['W2'] + params['b2'])
    final_concat = np.concatenate([gmf_vec, h2], axis=1)
    pred = (final_concat @ params['Wout'] + params['bout']).flatten()
    return pred + global_mean

best_val_rmse = np.inf
best_params = None
patience_counter = 0

n_train = len(train_u)
for epoch in range(N_EPOCHS_NCF):
    perm = np.random.permutation(n_train)
    epoch_loss = 0.0
    for start in range(0, n_train, BATCH_SIZE):
        t_step += 1
        batch_idx = perm[start:start + BATCH_SIZE]
        u_b, i_b, r_b = train_u[batch_idx], train_i[batch_idx], train_r[batch_idx] - global_mean
        bsz = len(u_b)

        # ---- Forward ----
        p_g, q_g = params['P_gmf'][u_b], params['Q_gmf'][i_b]
        gmf_vec = p_g * q_g                                   # (bsz, K_GMF)

        p_m, q_m = params['P_mlp'][u_b], params['Q_mlp'][i_b]
        mlp_in = np.concatenate([p_m, q_m], axis=1)           # (bsz, 2*K_MLP)
        z1 = mlp_in @ params['W1'] + params['b1']
        h1 = relu(z1)
        z2 = h1 @ params['W2'] + params['b2']
        h2 = relu(z2)

        final_concat = np.concatenate([gmf_vec, h2], axis=1)  # (bsz, K_GMF+H2)
        pred = (final_concat @ params['Wout'] + params['bout']).flatten()
        err = pred - r_b
        epoch_loss += np.sum(err ** 2)

        # ---- Backward ----
        d_pred = (2 * err / bsz).reshape(-1, 1)                # (bsz,1)
        grad_Wout = final_concat.T @ d_pred
        grad_bout = d_pred.sum(axis=0)
        d_final_concat = d_pred @ params['Wout'].T              # (bsz, K_GMF+H2)

        d_gmf_vec = d_final_concat[:, :K_GMF]
        d_h2 = d_final_concat[:, K_GMF:]

        d_z2 = d_h2 * relu_deriv(z2)
        grad_W2 = h1.T @ d_z2
        grad_b2 = d_z2.sum(axis=0)
        d_h1 = d_z2 @ params['W2'].T

        d_z1 = d_h1 * relu_deriv(z1)
        grad_W1 = mlp_in.T @ d_z1
        grad_b1 = d_z1.sum(axis=0)
        d_mlp_in = d_z1 @ params['W1'].T
        d_p_m, d_q_m = d_mlp_in[:, :K_MLP], d_mlp_in[:, K_MLP:]

        d_p_g = d_gmf_vec * q_g
        d_q_g = d_gmf_vec * p_g

        # Gộp gradient theo từng user/item index (vì có thể trùng trong batch)
        grad_P_gmf = np.zeros_like(params['P_gmf']); np.add.at(grad_P_gmf, u_b, d_p_g)
        grad_Q_gmf = np.zeros_like(params['Q_gmf']); np.add.at(grad_Q_gmf, i_b, d_q_g)
        grad_P_mlp = np.zeros_like(params['P_mlp']); np.add.at(grad_P_mlp, u_b, d_p_m)
        grad_Q_mlp = np.zeros_like(params['Q_mlp']); np.add.at(grad_Q_mlp, i_b, d_q_m)

        # Thêm L2 regularization (weight decay) lên embeddings để chống overfitting
        grad_P_gmf += REG_NCF * params['P_gmf']
        grad_Q_gmf += REG_NCF * params['Q_gmf']
        grad_P_mlp += REG_NCF * params['P_mlp']
        grad_Q_mlp += REG_NCF * params['Q_mlp']

        adam_update('Wout', grad_Wout); adam_update('bout', grad_bout)
        adam_update('W2', grad_W2); adam_update('b2', grad_b2)
        adam_update('W1', grad_W1); adam_update('b1', grad_b1)
        adam_update('P_gmf', grad_P_gmf); adam_update('Q_gmf', grad_Q_gmf)
        adam_update('P_mlp', grad_P_mlp); adam_update('Q_mlp', grad_Q_mlp)

    # ---- Đánh giá trên tập validation sau mỗi epoch (early stopping) ----
    val_pred = np.clip(ncf_predict(val_u, val_i), 1, 5)
    val_rmse = rmse(val_r, val_pred)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS_NCF} - Train MSE: {epoch_loss/n_train:.4f} - Val RMSE: {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-4:
        best_val_rmse = val_rmse
        best_params = {k: v.copy() for k, v in params.items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  --> Early stopping tại epoch {epoch+1} (Val RMSE tốt nhất: {best_val_rmse:.4f})")
            break

params = best_params  # khôi phục trọng số tốt nhất trên validation

ncf_preds = np.clip(ncf_predict(test_u, test_i), 1, 5)
ncf_rmse = rmse(test_r, ncf_preds)
ncf_mae = mae(test_r, ncf_preds)
ncf_time = time.time() - t0
print(f"RMSE: {ncf_rmse:.4f} | MAE: {ncf_mae:.4f} | Thời gian: {ncf_time:.1f}s")
results.append(('NCF (GMF + MLP) - Đề xuất', ncf_rmse, ncf_mae, ncf_time))

# ============================================================
# BẢNG SO SÁNH KẾT QUẢ (dùng cho mục 3.3.2)
# ============================================================
print("\n" + "="*60)
print("BẢNG SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH")
print("="*60)
result_df = pd.DataFrame(results, columns=['Mô hình', 'RMSE', 'MAE', 'Thời gian huấn luyện (s)'])
result_df['RMSE'] = result_df['RMSE'].round(4)
result_df['MAE'] = result_df['MAE'].round(4)
result_df['Thời gian huấn luyện (s)'] = result_df['Thời gian huấn luyện (s)'].round(1)
print(result_df.to_string(index=False))

result_df.to_csv('/home/claude/model_comparison_table.csv', index=False, encoding='utf-8-sig')
print("\nĐã lưu bảng so sánh: model_comparison_table.csv")
