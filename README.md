# Occluded plate detector
A software solution for automatic detection and reconstruction of vehicle license plates from images. The system is able to work with partially visible or overlapping license plates: it searches for similar cars in the dataset and finds the same car with a complete license plate.
### 1. Project Team
1.	Dmitry Baranov — Scrum Master (Documentation & Agile Process)
2.	Nail Siraev — Machine Learning Engineering (Model Architecture, Training & ML Ops)
3.	Shamil Gafiyatullin — Machine Learning Engineering (Model Architecture, Training & ML Ops)
4.	Maksim Pishchulin — Machine Learning Engineering (Model Architecture, Training & ML Ops)
5.	Daniil — 
6.	Maria Isakova — Frontend Development
7.	Daria Sabirova — Backend Development (API Server)
________________________________________
### 2. Technology Stack
•	Programming Language: Python
•	Computer Vision: YOLOv8, ResNet-108
•	OCR (распознавание текста): Tesseract
•	Deep Learning Framework: PyTorch
•	Backend: FastAPI / Flask
•	Data Processing: NumPy, Pandas
•	Database: ElasticSearch ________________________________________
### 3. Repository Structure
The repository follows standard software engineering practices for AI feature integration:
•	/specs/ — Product requirements and specifications (PRD, Data Spec, DoD).
•	/src/ — main application code (models, API, image processing logic).
•	/tests/ — test images (partial and full license plates).
•	/Notebooks/ — Jupyter notebooks for data analysis, exploration, and model experimentation.
________________________________________
### 3. Model Artifacts
________________________________________
### 4. System Architecture
The system is designed using a modular architecture:
Core Components:
• Image Input Module
Handles input image upload
• License Plate Detection Module
Detects the plate region
• OCR Module
Extracts text from the plate
• Completeness Check Module
Determines whether the plate is fully visible
• Similarity Search Module
Finds visually similar vehicles in the dataset
• Search Module
Outputs the 5 most similar images
• API Layer
Provides external access to the system
________________________________________
### 5. Execution Flow
 ________________________________________
### 6. Quick Start
________________________________________



# Детектор перекрытых номерных знаков (Occluded Plate Detector)
Occluded Plate Detector — это программное решение для автоматического обнаружения и восстановления автомобильных номерных знаков по изображениям.
Система способна работать с частично видимыми или перекрытыми номерами: она ищет похожие автомобили в датасете и находит такой же автомобиль с полным номерным знаком.
________________________________________
### 1. Project Team
1.	Dmitry Baranov — Scrum Master (Documentation & Agile Process)
2.	Nail Siraev — Machine Learning Engineering (Model Architecture, Training & ML Ops)
3.	Shamil Gafiyatullin — Machine Learning Engineering (Model Architecture, Training & ML Ops)
4.	Maksim Pishchulin — Machine Learning Engineering (Model Architecture, Training & ML Ops)
5.	Daniil —
6.	Maria Isakova — Frontend Development
7.	Daria Sabirova — Backend Development (API Server)
________________________________________
### 2. Technology Stack
• Язык программирования: Python
• Компьютерное зрение: YOLOv8, ResNet-108
• OCR (распознавание текста): Tesseract
• Фреймворк глубокого обучения: PyTorch
• Backend: FastAPI / Flask
• Обработка данных: NumPy, Pandas
• База данных: ElasticSearch
________________________________________
### 3. Repository Structure
Репозиторий организован в соответствии со стандартными практиками разработки ПО с использованием AI:
• /specs/ — требования к продукту и спецификации (PRD, Data Spec, DoD)
• /src/ — основной код приложения (модели, API, логика обработки изображений)
• /tests/ — тестовые изображения (частичные и полные номерные знаки)
• /Notebooks/ — Jupyter Notebook для анализа данных, исследований и экспериментов
________________________________________
### 3. Model Artifacts
________________________________________
### 4. System Architecture
Система построена по модульной архитектуре.
Основные компоненты:
• Модуль загрузки изображения (Image Input Module)
Обрабатывает загрузку входного изображения
• Модуль детекции номерного знака (License Plate Detection Module)
Определяет область номерного знака на изображении
• Модуль OCR
Извлекает текст с номерного знака
• Модуль проверки полноты (Completeness Check Module)
Определяет, полностью ли виден номер
• Модуль поиска похожих изображений (Similarity Search Module)
Находит визуально похожие автомобили в датасете
• Модуль поиска (Search Module)
Возвращает 5 наиболее похожих изображений
• API-слой (API Layer)
Обеспечивает внешний доступ к системе
________________________________________
### 5. Execution Flow
 
________________________________________
### 6. Quick Start
________________________________________

