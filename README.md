# Data Scientist 

## Technical Skills
- Programming languages: Java, Python
- Libraries and Frameworks: Keras, NL ​Toolkit, Numpy, Pandas, PySpark, PyTorch, Scikit-​learn, SciPy, Seaborn, TensorFlow
- Machine learning: A/​B ​Testing, Algorithms, Anomaly ​Detection, Artificial ​intelligence, BERT, Big ​Data, BPE ​tokenizer, Computational ​Linguistics, Conversational ​AI, Data ​Engineering, - Data ​Mining, Data ​Modeling, Debugging, Deep ​Learning ​Methods, End-​point ​Deployment, Ensemble ​Methods, Generative ​AI, Generative ​model, Hypotheses ​testing, K-​means ​clustering, langchain ​, Language ​Model ​, Linear ​regression, Llama2, Logistic ​regression, Neural ​Networks, Quantization, Speech ​Recognition, Statistics, Supervised, Support ​Vector ​Machine ​(​SVM)​, Tagging, Text ​Classification, Text-​to-​Speech, Time ​series ​signal, Tokenization, Transformer ​models, Unsupervised, Wav2vec, word2vec, Word ​Embeddings
- Cloud: API, AWS ​EC2, AWS ​EMR, AWS ​S3, AWS ​Sagemaker, Gradio
- Databases: MongoDB, MySQL, NoSQL, SQLite

## Education
- M.S., Data Science | The University of New Haven, May 2024
- B.S., Computer Science | The University of Kharazmi, September 2016

## Work Experience
### Graduate Research Assistant @ University of New Haven (October 2022 - May 2024)
- Integrated and deployed a Large Language Model (LLM) based on state-of-the-art TTS for low resource languages, achieving an exceptional Mean Opinion Score (MOS) of 4.2 out of 5 across multiple low resource languages.
- Automated key performance metrics of the low-resource TTS evaluation system based on 5 ontologies, resulting in 40% faster evaluation times and 25% accuracy improvement by fine-tuning the wav2vec model for low-resource languages.
- Enhanced AI/ML platform infrastructure and performance management by optimizing a large language model (LLM)-based text-to-speech system. Leveraged multi-GPU setups and advanced GPT training techniques, achieving a 20% faster convergence rate for the model
- Automated the creation of a high-quality Text-to-Speech (TTS) dataset for a low-resource language using the latest open-source data quality tools, improving accessibility and speech synthesis by 12%. This process enhanced user experience and broadened product reach.
![XTTS_Persian](/assets/XTTSSVG.png)

### Data Analyst @ Maadiran Industries Group, TCL Exclusive Home Appliance representative (December 2017 - January 2022)
- Developed and managed budgets to optimize business metrics, automated workflows, and monitored Key Performance Indicators (KPIs), resulting in a 25% increase in on-time delivery and $80,000 reduction in quarterly shipping costs.

- Improved technical and non-technical key stakeholders' decision-making efficiency by 20% through statistical methods, Python, analytics software, and data visualization tools, resulting in enhanced business strategy, provided technical solutions for business problems, and meaningful insights.
- Enhanced query performance through strategic optimization techniques, including setting efficient indexes, separating log files from data files, and implementing various other optimization methods.
- Effectively trained customers in financial systems, resulting in a significant increase in employees' knowledge and a 20% improvement in software usage efficiency.


## Projects
### NLP Multilingual Teacher Assistant Chatbot
![Pipeline](/assets/4957A332-F9E6-43AC-887E-43E3C30D907B.jpeg)
- Developed a scalable, real-time, API Gateway-integrated, multi-modal end-to-end data pipeline that handles speech prompts in Persian, English, and Hindi. Utilized OpenAI's embedding technology for creating Vector DB and the Whisper model for converting speech to text, enhancing data security and improving information retrieval.
- Implemented and deployed the entire pipeline to operate seamlessly across multiple languages and integrates robustly with Gradio, providing a scalable, cost-effective chatbot service that supports new features development and significantly drives growth.
- [Explore the Deployment video on Lambda GPU](https://www.youtube.com/watch?v=1M_WZ35WaPs)
  
### Freight Forecasting and Delivery Schedule Optimization
Feature Engineering and Model Development:

- Conducted extensive feature engineering and extraction on unstructured data, preparing it for machine learning models through data wrangling and cleaning.
- Applied three machine learning algorithms: ARIMA and LSTM for freight forecasting, and a Dense Neural Network (DNN) for optimizing delivery schedules. This included designing and implementing the model architectures, training the models, and evaluating their performance.
  
Performance Improvement and Optimization:

- Achieved significant performance improvements in forecasting monthly freight costs and optimizing delivery schedules, surpassing naive benchmarks and enhancing prediction accuracy.
- Optimized the delivery schedule by accurately predicting deviations between scheduled and actual delivery times, leading to more efficient logistics and operational planning.
#### Summary of Findings

| **Model** | **Objective**                         | **Target Variable**        | **Model Architecture**          | **Performance Metric** | **Training RMSE** | **Test RMSE**      | **Benchmark RMSE** |
|-----------|---------------------------------------|----------------------------|---------------------------------|------------------------|-------------------|--------------------|--------------------|
| DNN       | Optimize delivery schedules           | Schedule v Actual          | Dense Neural Network (4 layers) | RMSE                   | 22.86             | 25.88              | 29.22            |
| LSTM      | Forecast monthly freight costs        | Freight Cost (USD)         | LSTM Network (1 LSTM layer)     | RMSE                   | 292287.38 | 529783.7       | 861773.48        |
| ARIMA     | Forecast monthly freight costs        | Freight Cost (USD)         | ARIMA (2,1,2)                   | RMSE                   | 323737.57         | 565322.23          | 861773.48        |
  
### Anomaly Detection in Time Series Medical devices Signal
![Laparoscopic Stapler fault signal](/assets/fault.jpeg)
- Enhanced medical device (surgical staplers) failure prediction by 25% and decreased operational downtime by 20% by integrating machine learning algorithms. These improvements led to more accurate equipment failure predictions and reduced operational interruptions.
- Determined the most effective generative model for fault detection by assessing Variational Autoencoder (97%), Generative Adversarial Network (95%), and Hidden Markov Model (82%), resulting in enhanced fault detection in 6 months.
  
### Automated Chest X-Ray Analysis for Pneumonia Detection
- Enhanced image processing algorithm to extract nm level information from scientific imaging equipment, improving precision by 20%, optimizing engineering processes.
- Developed a user-friendly web interface for real-time medical image analysis using computer vision techniques, facilitating rapid assessment and treatment planning, contributing to potentially improved patient outcomes. Integrated layering techniques in image processing to refine diagnostic tools.
- [Explore the Hugging Face Space](https://huggingface.co/spaces/barghavani/chest_x_ray_diagnose)
  
### Malware classification of PE Files by help of Feature Extraction and Deep learning models
- Deployed a PE file classification model on AWS SageMaker for easy client access, incorporating DevOps practices for simplified interaction with the model and ensuring a seamless user experience.
- Developed and deployed forecasting modeling using CNN and Random Forest classifiers on unstructured large-scale datasets (PE files) to achieve 95% accuracy. Created a dashboard for monitoring model performance and accuracy.
- [Explore the Deployment video on AWS](https://youtu.be/q6CPYSwuuUM)
  
### Resume-analytics
- Developed scalable AI algorithms and models, leading to a 40% improvement in model deployment time and enhancing overall business performance.
- Spearheaded the development of an AI-powered resume parsing tool utilizing OpenAI's GPT-3.5 model in 5 days.
- Developed and implemented a user-friendly web interface with Gradio to facilitate seamless resume parsing. Utilized advanced construction methods to optimize code generation and content creation, ensuring alignment with project objectives and providing actionable recommendations for improvement.
- [Explore the Hugging Face Space](https://huggingface.co/spaces/barghavani/Resume_ATS)


## Awards & Scholarship
- 2023-2024 TCoE Endowed Graduate Fellowship, recognizing excellence in technology and engineering research.

## Publications
- Comparative Study of Generative Models for Early Detection of Failures in Medical Devices, Conference ICMHI 2024.

## Volunteering  & Leadership 
- Data science club University of New Haven

## About
Experienced Machine Learning (ML) engineer with over 5 years of expertise in data analysis, optimization, and research in Generative AI. Continuously learning to stay abreast of the latest trends in data science. Notable projects include developing generative models for fault detection in medical devices, achieving 97% accuracy, and enhancing a multilingual transformer-based Text-to-Speech (TTS) system for low-resource languages, which achieved a mean opinion score of 4.2 out of 5 in 6 months through innovative approaches. Led brainstorming sessions to identify new opportunities in testing automation, reducing testing time by 40%. Implemented and maintained a speech conversation chatbot, improving user experience by 40% through effective MLOps deployment and management.
## Contact Information
- **Mobile:** +1-860-944-5353
- **Email:** [bahareh.arghavani@gmail.com](mailto:bahareh.arghavani@gmail.com)
- **LinkedIn:** [Bahareh Arghavani Nobar](https://www.linkedin.com/in/bahareh-arghavan/)
- **Hugging Face Space:** [Visit my Hugging Face Space](https://huggingface.co/barghavani)

Feel free to reach out for collaborations or inquiries!
