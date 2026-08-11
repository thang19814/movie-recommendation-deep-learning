"""
CHƯƠNG 3: MÔ HÌNH THỰC NGHIỆM
3.1. Tiền xử lý và Phân tích khám phá dữ liệu (EDA)
3.1.1. Làm sạch và chuẩn hóa dữ liệu từ file CSV
3.1.2. Trực quan hóa phân phối lượt đánh giá và thể loại phim được ưa chuộng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# ============================================================
# 3.1.1. LÀM SẠCH VÀ CHUẨN HÓA DỮ LIỆU TỪ FILE CSV
# ============================================================

# --- Bước 1: Đọc dữ liệu rating (u.data) ---
ratings = pd.read_csv(
    '/mnt/user-data/uploads/u.data',
    sep='\t',
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    encoding='latin-1'
)

# --- Bước 2: Đọc dữ liệu metadata phim ---
movies = pd.read_csv('/mnt/user-data/uploads/movielens_100k.csv', encoding='utf-8')

# --- Bước 3: Join hai bảng theo movie_id ---
df = pd.merge(ratings, movies, on='movie_id', how='inner')
print(f"[1] Số lượt đánh giá ban đầu: {len(ratings):,}")
print(f"[2] Số lượt đánh giá sau khi join: {len(df):,}")
print(f"[3] Số lượt đánh giá bị loại (thiếu metadata): {len(ratings) - len(df)}")

# --- Bước 4: Chuyển timestamp sang datetime ---
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# --- Bước 5: Xử lý dữ liệu thiếu (NaN) ở các cột metadata ---
for col in ['directors', 'actors', 'genres']:
    n_missing = df[col].isna().sum()
    df[col] = df[col].fillna('Unknown')
    print(f"[4] Cột '{col}': điền {n_missing:,} giá trị thiếu bằng 'Unknown'")

# --- Bước 6: Kiểm tra và loại bỏ trùng lặp ---
n_dup = df.duplicated(subset=['user_id', 'movie_id']).sum()
print(f"[5] Số cặp (user_id, movie_id) trùng lặp: {n_dup}")

# --- Bước 7: Tách chuỗi genres thành list (phục vụ phân tích) ---
df['genres_list'] = df['genres'].apply(lambda x: x.split() if x != 'Unknown' else ['Unknown'])

# --- Bước 8: Mã hóa lại user_id và movie_id thành index liên tục (0..N-1) ---
# Cần thiết cho tầng Embedding của mô hình NCF
user_ids = df['user_id'].unique().tolist()
movie_ids = df['movie_id'].unique().tolist()
user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
df['user_idx'] = df['user_id'].map(user2idx)
df['movie_idx'] = df['movie_id'].map(movie2idx)

n_users = len(user_ids)
n_movies = len(movie_ids)
print(f"[6] Số user sau mã hóa: {n_users}")
print(f"[7] Số movie sau mã hóa: {n_movies}")

# --- Bước 9: Tính độ thưa của ma trận (Sparsity) ---
n_ratings = len(df)
sparsity = 1 - (n_ratings / (n_users * n_movies))
print(f"[8] Độ thưa ma trận User-Item (Sparsity): {sparsity*100:.4f}%")
print(f"    (Chỉ {100-sparsity*100:.4f}% ô trong ma trận {n_users}x{n_movies} có giá trị)")

# Lưu dữ liệu đã làm sạch để dùng cho bước huấn luyện mô hình
df.to_csv('/home/claude/movielens_cleaned.csv', index=False)
print(f"\n[9] Đã lưu dữ liệu sạch: movielens_cleaned.csv ({df.shape[0]:,} dòng, {df.shape[1]} cột)")

# ============================================================
# 3.1.2. TRỰC QUAN HÓA PHÂN PHỐI LƯỢT ĐÁNH GIÁ VÀ THỂ LOẠI
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Biểu đồ 1: Phân phối điểm rating ---
rating_counts = df['rating'].value_counts().sort_index()
axes[0, 0].bar(rating_counts.index, rating_counts.values, color='#4C72B0', edgecolor='black')
axes[0, 0].set_title('Phân phối điểm đánh giá (Rating)', fontweight='bold')
axes[0, 0].set_xlabel('Điểm rating (sao)')
axes[0, 0].set_ylabel('Số lượt đánh giá')
for i, v in zip(rating_counts.index, rating_counts.values):
    axes[0, 0].text(i, v + 500, f'{v:,}', ha='center', fontsize=9)

# --- Biểu đồ 2: Top 10 thể loại phim được đánh giá nhiều nhất ---
all_genres = df.explode('genres_list')
genre_counts = all_genres[all_genres['genres_list'] != 'Unknown']['genres_list'].value_counts().head(10)
axes[0, 1].barh(genre_counts.index[::-1], genre_counts.values[::-1], color='#DD8452', edgecolor='black')
axes[0, 1].set_title('Top 10 thể loại phim được đánh giá nhiều nhất', fontweight='bold')
axes[0, 1].set_xlabel('Số lượt đánh giá')

# --- Biểu đồ 3: Số lượt đánh giá trên mỗi người dùng (phân phối) ---
ratings_per_user = df.groupby('user_id').size()
axes[1, 0].hist(ratings_per_user, bins=50, color='#55A868', edgecolor='black')
axes[1, 0].set_title('Phân phối số lượt đánh giá / người dùng', fontweight='bold')
axes[1, 0].set_xlabel('Số lượt đánh giá')
axes[1, 0].set_ylabel('Số người dùng')
axes[1, 0].axvline(ratings_per_user.median(), color='red', linestyle='--',
                    label=f'Trung vị: {ratings_per_user.median():.0f}')
axes[1, 0].legend()

# --- Biểu đồ 4: Số lượt đánh giá trên mỗi phim (phân phối) ---
ratings_per_movie = df.groupby('movie_id').size()
axes[1, 1].hist(ratings_per_movie, bins=50, color='#C44E52', edgecolor='black')
axes[1, 1].set_title('Phân phối số lượt đánh giá / phim', fontweight='bold')
axes[1, 1].set_xlabel('Số lượt đánh giá')
axes[1, 1].set_ylabel('Số phim')
axes[1, 1].axvline(ratings_per_movie.median(), color='red', linestyle='--',
                    label=f'Trung vị: {ratings_per_movie.median():.0f}')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('/home/claude/eda_overview.png', dpi=150, bbox_inches='tight')
print("\n[10] Đã lưu biểu đồ tổng quan: eda_overview.png")

# --- Bảng thống kê tóm tắt (cho vào báo cáo) ---
print("\n" + "="*50)
print("BẢNG THỐNG KÊ TÓM TẮT (dùng cho Chương 3 báo cáo)")
print("="*50)
summary = pd.DataFrame({
    'Chỉ số': [
        'Tổng số lượt đánh giá',
        'Số người dùng',
        'Số phim',
        'Điểm rating trung bình',
        'Độ thưa ma trận (Sparsity)',
        'Số lượt đánh giá TB / người dùng',
        'Số lượt đánh giá TB / phim'
    ],
    'Giá trị': [
        f"{n_ratings:,}",
        f"{n_users:,}",
        f"{n_movies:,}",
        f"{df['rating'].mean():.3f}",
        f"{sparsity*100:.3f}%",
        f"{ratings_per_user.mean():.1f}",
        f"{ratings_per_movie.mean():.1f}"
    ]
})
print(summary.to_string(index=False))
summary.to_csv('/home/claude/eda_summary_table.csv', index=False)
