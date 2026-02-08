# app.py
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import uuid
import shutil

app = Flask(__name__)
app.secret_key = 'supersecretkey123'  # Change ça pour une vraie clé secrète en production

# Configuration des dossiers
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Racine du projet
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
app.config['MODEL_FOLDER'] = os.path.join(BASE_DIR, 'model')
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250 Mo max

# Création des dossiers s’ils n’existent pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# === CHEMIN VERS TON MODÈLE LOCAL ===
MODEL_FILENAME = 'best.pt'
MODEL_PATH = os.path.join(app.config['MODEL_FOLDER'], MODEL_FILENAME)

# Chargement du modèle au démarrage
model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Le modèle n'est pas trouvé à : {MODEL_PATH}\n"
                               f"1. Télécharge best.pt depuis Colab\n"
                               f"2. Place-le dans le dossier 'model/' du projet")
    
    model = YOLO(MODEL_PATH)
    print(f"Modèle chargé avec succès depuis : {MODEL_PATH}")
except Exception as e:
    print("ERREUR CRITIQUE - Impossible de charger le modèle :")
    print(e)
    print("\nL'application va continuer mais les détections ne fonctionneront pas.")

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if model is None:
        flash("Le modèle n'est pas chargé. Vérifiez le chemin du fichier best.pt")
        return redirect(url_for('index'))

    if 'video' not in request.files:
        flash('Aucun fichier sélectionné')
        return redirect(url_for('index'))

    file = request.files['video']

    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Format non supporté. Formats acceptés : mp4, avi, mov, mkv, wmv')
        return redirect(url_for('index'))

    # Sauvegarde de la vidéo uploadée avec un nom unique
    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
    file.save(input_path)

    # Chemin de la vidéo de sortie
    output_filename = f"detected_{unique_id}_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    try:
        print(f"Début du traitement de la vidéo : {filename}")
        
        # Prédiction YOLOv8 sur la vidéo
        results = model.predict(
            source=input_path,
            conf=0.40,          # Seuil de confiance (tu peux ajuster)
            iou=0.55,
            save=True,          # Sauvegarde automatique
            project=app.config['OUTPUT_FOLDER'],
            name=unique_id,
            exist_ok=True,
            vid_stride=1,       # Traite toutes les frames (2 = une sur deux pour accélérer)
            line_width=2
        )

        # Ultralytics crée un sous-dossier → on récupère et déplace la vidéo générée
        predicted_dir = os.path.join(app.config['OUTPUT_FOLDER'], unique_id)
        predicted_video = os.path.join(predicted_dir, filename)

        if os.path.exists(predicted_video):
            shutil.move(predicted_video, output_path)
            # Nettoyage du dossier temporaire
            shutil.rmtree(predicted_dir, ignore_errors=True)
        else:
            raise FileNotFoundError("La vidéo annotée n'a pas été générée par YOLO")

        return render_template('result.html',
                             original_name=filename,
                             output_video=output_filename,
                             message="Détection terminée avec succès !")

    except Exception as e:
        flash(f"Erreur pendant le traitement de la vidéo : {str(e)}")
        # Nettoyage en cas d'erreur
        if os.path.exists(input_path):
            os.remove(input_path)
        return redirect(url_for('index'))

@app.route('/outputs/<filename>')
def get_output(filename):
    """Permet de servir les vidéos de sortie"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=False)

if __name__ == '__main__':
    print("Démarrage de l'application Flask...")
    print(f"Modèle utilisé : {MODEL_PATH}")
    app.run(debug=True, host='0.0.0.0', port=5000)