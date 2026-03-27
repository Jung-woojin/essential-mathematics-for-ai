# 인공지능 수학 Deepdive 코드

본 레포지토리는 인공지능 수학 심화 과정을 위한 코드와 해설을 담은 레포입니다.

**수학 이론과 코드의 연결을 통해 머신러닝/딥러닝의 본질을 이해하세요!**

---

## 📚 레포지토리 구조

```
essential-mathematics-for-ai/
├── README.md                    # 이 문서
├── requirements.txt             # dependencies
├── chapter01/                   # 소개 및 기본 수학 (추후 추가 예정)
├── chapter02/
│   ├── README.md                # ✅ 확률분포와 추정
│   ├── chapter2_distributions.ipynb
│   ├── chapter2_fitting_functions.ipynb
│   └── data/
├── chapter03/
│   ├── README.md                # ✅ 최적화
│   ├── chapter3.ipynb
│   └── data/
├── chapter04/
│   ├── README.md                # ✅ 행렬 분해
│   ├── chapter4.ipynb
├── chapter05/
│   ├── README.md                # ✅ 선형대수 & CV 응용
│   ├── chatper5.ipynb
│   ├── Fig_Gabor_formulas.svg
│   └── data/
└── chapter06/
    ├── README.md                # ✅ 심화 주제 & 실전 응용
    ├── chapter6.ipynb
    └── data/
```

---

## 🎯 목표

1. **수학 이론 이해**: 핵심 공식을 직관적으로 이해
2. **코드 구현**: NumPy, PyTorch 로 직접 구현
3. **실습**: Jupyter Notebook 으로 시각화 및 실험
4. **연계**: 이론 ↔ 코드 ↔ 실험 연결

---

## 📖 챕터별 개요

### Chapter 2: 확률분포와 추정
- 정규분포, 이항분포, 포아송분포
- 다변량 정규분포
- 함수 적합 (Least Squares, Regularization)
- 최대우도추정 (MLE)

**핵심 Jupyter:**
- `chapter2_distributions.ipynb` - 확률분포 시각화
- `chapter2_fitting_functions.ipynb` - 함수 적합 실습

---

### Chapter 3: 최적화
- 경사하강법 (Gradient Descent)
- 변형들 (Mini-batch, Momentum, Adam)
- 학습률 스케줄링
- 최적화 알고리즘 비교

**핵심 Jupyter:**
- `chapter3.ipynb` - 최적화 알고리즘 구현 및 비교

---

### Chapter 4: 행렬 분해
- LU 분해
- QR 분해
- **SVD (특이값 분해)** - 이미지 압축 등
- 대각화
- Cholesky 분해

**핵심 Jupyter:**
- `chapter4.ipynb` - 행렬 분해 및 응용

---

### Chapter 5: 선형대수 심화 & CV 응용
- 가우시안 필터, 그라디언트
- 라플라시안, LoG
- 가버 필터 (Gabor Filter)
- Receptive Field 분석

**핵심 Jupyter:**
- `chapter5.ipynb` - 컴퓨터비전 필터 구현

---

### Chapter 6: 심화 주제 & 실전 응용
- 베이지안 통계
- 정규화와의 연결 (Ridge/Lasso)
- 확률적 경사하강법 (SGD)
- 과적합 방지 전략

**핵심 Jupyter:**
- `chapter6.ipynb` - 실전 최적화 전략

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# 의존성 설치
pip3 install -r requirements.txt

# Jupyter 노트북 실행
jupyter notebook
```

### 2. 학습 가이드

각 챕터의 `README.md` 를 참고하세요:
- 핵심 개념 설명
- 수식과 코드 연결
- 실습 가이드
- 추가 학습 자료

### 3. Jupyter 노트북 실행

각 챕터별로 준비된 Jupyter 노트북을 실행하여:
- **시각화**: 수식과 개념 시각적 이해
- **코드 실행**: 직접 코드 수정 및 실험
- **실습**: 연습문제 해결

---

## 🔧 주요 라이브러리

- **NumPy**: 수치 계산, 행렬 연산
- **SciPy**: 과학기술 컴퓨팅
- **Matplotlib**: 시각화
- **scikit-learn**: 머신러닝 알고리즘
- **PyTorch**: 딥러닝 프레임워크

---

## 📊 추천 학습 경로

1. **기본:** Chapter 2 → Chapter 3
2. **심화:** Chapter 4 → Chapter 5
3. **실전:** Chapter 6
4. **복습:** 각 챕터 Jupyter 노트북 실행

---

## 🎓 학습 목표 달성 후

- ✅ 수학적 배경 없이 이해하기 어려웠던 개념들 파악
- ✅ 최적화 알고리즘 선택 및 튜닝 능력 향상
- ✅ 행렬 분해의 실전 활용 (차원축소, 압축 등)
- ✅ 컴퓨터비전 필터링 기법 이해
- ✅ 과적합 방지 및 모델 정규화 전략 수립

---

## 🤝 기여

본 레포지토리는 교육용 자료입니다. 개선 사항을 발견하셨다면 issue 나 PR 를 환영합니다!

---

## 📝 라이선스

MIT License

---

*최종 수정일: 2026 년 3 월*
*Created with 💜 for machine learning enthusiasts*
