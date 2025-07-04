from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import re
import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import requests
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
MODEL_NAME = "abdmuffid/fine-tuned-indo-sentiment-3-class"
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "deepseek-r1-distill-llama-70b"

pdf_text_global = ""
def download_model_once(model_name):
    try:
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSequenceClassification.from_pretrained(model_name)
        print("Model dan tokenizer sudah ada di cache atau berhasil di-download")
    except Exception as e:
        print(f"Gagal download model/tokenizer: {e}")

# Download model dan tokenizer sekali saat startup
download_model_once(MODEL_NAME)

# Load pipeline tanpa local_files_only=True agar bisa auto-download jika belum ada
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME
)

@app.route('/sentimen', methods=['POST'])
def sentimen():
    data = request.json
    print("Data diterima:", data)  # Debug print

    ulasan = data.get('ulasan', '')
    if not ulasan:
        return jsonify({"error": "ulasan kosong"}), 400

    try:
        result = sentiment_analysis(ulasan)
        label = result[0]['label']
        score = result[0]['score']

        return jsonify({
            "class": label,
            "score": score
        })
    except Exception as e:
        print("Error saat analisis:", str(e))  # Debug print
        return jsonify({"error": str(e)}), 500

@app.route('/helo')
def hello():
    return jsonify({"message": "Hellsso from Flask!"})


def bersihkan_teks(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:?!-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route('/terima-data', methods=['POST'])
def terima_data():
    ulasan_data = request.json.get('ulasan_data', [])
    umkm_data = request.json.get('umkmData', [])

    df_ulasan = pd.DataFrame(ulasan_data)
    df_umkm = pd.DataFrame(umkm_data)

    # Cek id_umkm tersedia
    if 'id_umkm' not in df_umkm.columns:
        return jsonify({'error': 'Kolom id_umkm tidak ditemukan pada data UMKM'}), 400

    df_ulasan['ringkasan_umkm'] = df_ulasan['ringkasan_umkm'].apply(bersihkan_teks)
    df_umkm['ringkasan_umkm'] = df_umkm['ringkasan_umkm'].apply(bersihkan_teks)

    def gabungkan_kolom(row):
        return f"{row.get('nama_umkm','')} {row.get('ringkasan_umkm','')} {row.get('produk','')}"
    
    df_ulasan['teks_gabungan'] = df_ulasan.apply(gabungkan_kolom, axis=1)
    df_umkm['teks_gabungan'] = df_umkm.apply(gabungkan_kolom, axis=1)

    semua_teks = pd.concat([df_ulasan['teks_gabungan'], df_umkm['teks_gabungan']]).reset_index(drop=True)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(semua_teks)

    tfidf_ulasan = vectorizer.transform(df_ulasan['teks_gabungan'])
    tfidf_umkm = vectorizer.transform(df_umkm['teks_gabungan'])

    cosine_sim_matrix = cosine_similarity(tfidf_ulasan, tfidf_umkm)

    top_n = 5
    rekomendasi_list = []
    umkm_terpakai = set()  # untuk melacak UMKM yang sudah direkomendasikan

    for i in range(cosine_sim_matrix.shape[0]):
        sorted_indices = cosine_sim_matrix[i].argsort()[::-1]  # urutkan dari tertinggi ke terendah

        rekomendasi_umkm = []
        for idx in sorted_indices:
            id_umkm = df_umkm.iloc[idx]['id_umkm']
            if id_umkm not in umkm_terpakai:
                umkm_terpakai.add(id_umkm)
                rekomendasi_umkm.append({
                    'id_umkm': id_umkm,
                    'nama_umkm': df_umkm.iloc[idx]['nama_umkm'],
                    'ringkasan_umkm': df_umkm.iloc[idx]['ringkasan_umkm'],
                    'produk': df_umkm.iloc[idx]['produk']
                })
            if len(rekomendasi_umkm) == top_n:
                break
        
        rekomendasi_list.append({
            'ulasan': df_ulasan.iloc[i][['nama_umkm', 'ringkasan_umkm', 'produk']].to_dict(),
            'rekomendasi_umkm': rekomendasi_umkm
        })

    return jsonify({
        'status': 'diterima',
        'jumlah_ulasan': len(ulasan_data),
        'jumlah_umkm': len(umkm_data),
        'rekomendasi': rekomendasi_list
    })

@app.route('/anomali', methods=['POST'])
def deteksi_anomali():
    data = request.get_json()

    # Validasi awal
    if not data or not isinstance(data, list):
        return jsonify({'error': 'Data harus berupa list transaksi'}), 400

    df = pd.DataFrame(data)

    # Ubah kolom 'kategori' jadi 'tipe' agar konsisten
    if 'kategori' in df.columns:
        df['tipe'] = df['kategori']
    else:
        return jsonify({'error': 'Kolom kategori tidak ditemukan'}), 400

    # Konversi ke numerik
    df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce')
    df = df.dropna(subset=['jumlah'])

    # Inisialisasi status default
    df["status"] = "✅ Aman"

    # Aturan global: terlalu kecil atau terlalu besar
    df.loc[df["jumlah"] < 10000, "status"] = "🧐 Perlu Diaudit"
    df.loc[df["jumlah"] > 10000000, "status"] = "⚠ Warning"

    # Filter untuk deteksi outlier berdasarkan tipe
    df_pengeluaran = df[
        (df["jumlah"] >= 10000) & 
        (df["jumlah"] <= 10000000)
    ]

    for tipe in df_pengeluaran["tipe"].unique():
        group = df_pengeluaran[df_pengeluaran["tipe"] == tipe]
        
        if len(group) < 3:
            fallback_idx = group[group["jumlah"] < 100000].index
            df.loc[fallback_idx, "status"] = "🧐 Perlu Diaudit"
            continue

        Q1 = group["jumlah"].quantile(0.25)
        Q3 = group["jumlah"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_idx = group[
            (group["jumlah"] < lower) | 
            (group["jumlah"] > upper)
        ].index
        df.loc[outlier_idx, "status"] = "🧐 Perlu Diaudit"

    return jsonify(df.to_dict(orient="records"))

def remove_think_tags(text):
    # Hapus tag <think> beserta isinya (multiline)
    return re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()

def query_model(chat_history):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": chat_history
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if resp.status_code == 200:
        result = resp.json()
        raw_content = result['choices'][0]['message']['content']
        clean_content = remove_think_tags(raw_content)
        return clean_content
    else:
        try:
            err = resp.json()
        except:
            err = resp.text
        return f"Terjadi kesalahan: {err}"

@app.route('/api/upload_pdf', methods=['POST'])
def upload_pdf():
    global pdf_text_global
    if 'pdf_file' not in request.files:
        return jsonify({"error": "Tidak ada file PDF yang diupload"}), 400

    pdf_file = request.files['pdf_file']
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages = [page.extract_text() or '' for page in pdf_reader.pages]
        pdf_text_global = "\n".join(pages).strip()

        if not pdf_text_global:
            return jsonify({"error": "File PDF kosong atau tidak bisa dibaca"}), 400

        chat_history = [
            {"role": "system", "content": "Anda adalah asisten yang merangkum isi dokumen secara singkat dan jelas."},
            {"role": "user", "content": f"Tolong buat ringkasan dari teks berikut:\n{pdf_text_global}"}
        ]
        response_text = query_model(chat_history)
        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": f"Gagal membaca file PDF: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    global pdf_text_global
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({"error": "Prompt tidak boleh kosong"}), 400

    chat_history = [
     {
  "role": "system",
  "content": "Sebagai admin yang berbahasa Indonesia, saya memiliki peran penting dalam mengelola chatbot Transparansi Dana Desa agar masyarakat bisa mengecek alokasi dana desanya dengan mudah dan transparan. Saya dapat mengelola data alokasi dana dengan cara memasukkan, memperbarui, atau menghapus data berdasarkan tahun, bidang penggunaan, dan jumlah dana yang tersedia di desa.\n\nSaya juga bertanggung jawab untuk memverifikasi dan memvalidasi setiap data yang masuk dari petugas atau perangkat desa, agar informasi yang ditampilkan benar-benar akurat dan dapat dipercaya oleh masyarakat. Dalam hal komunikasi, saya dapat mengatur isi chatbot, menambahkan kata kunci baru, menyesuaikan jawaban, serta memastikan chatbot menjawab dengan sopan, jelas, dan sesuai konteks waktu.\n\nSebagai admin, saya juga dapat memantau aktivitas pengguna yang berinteraksi dengan chatbot, melihat riwayat pertanyaan, serta menganalisis topik-topik yang paling sering ditanyakan. Informasi ini sangat membantu untuk mengetahui apa yang sedang dibutuhkan masyarakat dan meningkatkan pelayanan informasi publik.\n\nSaya juga memiliki akses ke laporan dan statistik penggunaan chatbot yang bisa diunduh secara mingguan atau bulanan. Selain itu, chatbot dapat diintegrasikan dengan sistem administrasi desa agar data yang ditampilkan selalu diperbarui secara otomatis. Tidak hanya itu, saya bisa mengatur pesan-pesan edukatif yang dikirimkan secara berkala kepada warga untuk meningkatkan pemahaman mereka mengenai pengelolaan dana desa dan dampaknya bagi pembangunan desa."
}
    ]

    if pdf_text_global:
        chat_history.append({"role": "system", "content": f"Berikut isi dokumen PDF yang sudah diupload:\n{pdf_text_global}"})
    chat_history.append({"role": "user", "content": prompt})

    response_text = query_model(chat_history)
    return jsonify({"response": response_text})
if __name__ == '__main__':
    # Jalankan Flask dengan debug mode aktif untuk tahu error lebih jelas
    app.run(port=5000, debug=True)