# Detecting Abnormal Sound for Industrial Machinery
---
  (산업기계 이상음 탐지를 위한 데이터 다운 샘플링 기법 비교 및 머신러닝 방법 제안) 
--- 

> 최근 인공지능의 빠른 발전으로 다양한 산업 분야에서 활용 연구가 활발하다. 그중 산업 기계에서 발생하는 이
상음 탐지 연구를 진행하였다. 음성 데이터는 머신러닝에 이용하기 전에 많은 전처리 작업을 통해 재가공 된
다. 그중 음성 데이터를 다운 샘플링 할 때에 고려할 수 있는 기존의 음성 신호처리 기법들과 음성 데이터에 적
용 사례가 없던 맥스 풀링 기법을 비교 실험하였다. 이때 비교한 모든 다운 샘플링 기법에서 공통적으로 성능
이 좋았던 다운 샘플링 비율 또한 관찰되었다. 최종적으로 본 논문에서는 머신러닝을 통한 기계 이상음 탐지를
진행할 때, 음성 데이터의 feature(특징)가 가장 잘 표현되는 맥스 풀링 기법과 최적의 다운 샘플링 비율을 제안한다.

## Dataset
> 본 연구에서 사용된 데이터 셋은 오작동하는 산업용 기계 조사및 검사를 위한 사운드 데이터 셋으로, 밸브, 펌프, 팬, 슬라이드 레일 4가지이며, 각 모델의 종류마다 4개의 다른 모델로부터 수집된 음성 파일이다.
> 
> 음성 파일은 16kHz 샘플링 속도와 16비트의 8채널 마이크로 녹음되어, 3가지의 소음이 섞여 소음 별로 배포되었다.
<div align="center">
 
||정상|비정상|합계|
|:-----:|:-----:|:-----:|:-----:|
|Fan 00(noise 0)| 1011 | 407 | 1418 |
|Fan 02(noise 0)| 1016 | 359 | 1375 |
|Fan 04(noise 0)| 1033 | 348 | 1381 |
|Fan 06(noise 0)| 1015 | 361 | 5550 |
| 합계 | 1015 | 361 | 5550 |

</div>
> 다음의 사진은  데이터에서 정상과 비정상 데이터를
무작위로 뽑아 시각화한 예시이다

<div align="center">

![데이터예시](https://github.com/SeungW/Roblox-Play-Video-Text-Analysis/assets/108673913/626a15f1-0648-4bd5-ace3-bd0740fb4365)

</div>

## 분석방법
> 기존 음성 데이터의 다운 샘플링 기법은 파이썬 라이브러리인
“librosa”에 소개되어 있으며, MaxPooling을 제외한
총 15개이다. 이 방법들은 모두 신호 처리에서 사용되는 수리
모델로서, 초당 Hz대역으로 샘플링 된 음성 데이터를 다운
샘플링한다. 가령, 16kHz 대역으로 수집된 데이터는 𝑋 ∈
ℝ16,000×𝑆 이다. 𝑆 는 음성 파일의 초를 의미하는데 librosa
라이브러리를 이용하면, 𝑋′ ∈ ℝ𝑁×𝑆로 데이터의 길이를 줄일
수 있다. 하이퍼 파라미터 𝑁은 사용자가 16,000보다 작은 임의의
수로 설정이 가능하다. 이때 다운 샘플링 되는 속도와 처리되는
데이터의 질은 trade-off 관계에 있으며, 15개의 방법 각각은
처리 속도와 데이터의 질을 선택적으로 고려한 방법들의
차이이다.
>
### 맥스풀링 방법
> 아날로그에서 디지털 신호로 수집된 음성 데이터 𝑋 ∈ ℝ𝐻𝑧×𝑆은
그 주파수가 Hz 대역에 따라 벡터로 표현되었다. 이때 이상음이
포함된 음성 데이터 𝑋에서 저주파 혹은 고주파의 신호는 이상치
탐지 분야 특성 상 중요한 신호일 것으로 판단된다.

<div align="center">
  
![MaxPooling](https://github.com/SeungW/Roblox-Play-Video-Text-Analysis/assets/108673913/d328807a-42e7-4a73-ac5b-c5032ed5ae68)

</div>

---

## 최적의 다운샘플링 비율
> 데이터를 다운 샘플링 할 때에, 크게 고려되는 사항은 하이퍼
파라미터 𝑁이다. 이는 사용자가 지정해야 하는 사항이지만,
머신러닝에서 최적의 성능이 도출되는 값을 어떠한 정보 없이
선택하기란 쉽지 않다. 
> 
> 음성 파일 당 10초, 16kHz 대역인 데이터를 다운 샘플링할 때에, 하이퍼
파라미터 𝑁에 대해 원본 데이터의 hz길이인 160,000의 약수를
근거하여 총 21개의 𝑁 = {64, 80, 100,125, 128, 160, 200, 250, 320,
400, 500, 640, 800, 1000, 1600, 2000, 3200, 4000, 8000, 10000,
13000}을 설정하여, 각 성능을 비교 실험하였다.

--- 

## 학습 및 추론 
> 데이터 셋은 각 모델의 종류 마다 층화 추출법을 통해 8 : 2로
Train set, Test set 구성하였다. 인공지능 모델은 부스팅 트리
모델 중 하나로써 수직적으로 트리를 확장하여 빠른 속도로
loss를 줄이는 Light GBM[3]을 사용하였다. 총 16개의 기법에
대해 21개의 다운 샘플링 비율인 𝑁 의 개수만큼 성능을
측정하였다. 이때 성능 평가 지표는 불균형 데이터에서 많이
쓰이는 f1 score 수식으로 비교 분석한다.

$$f1-score=2×\frac{precision×recall}{precision+recall}$$

---
## 결과
> 각 기법과 다운 샘플링 비율의 성능을 비교한
플롯이다. Fan 00모델의 성능을 제외하고는 MaxPooling 기법이
압도적인 성능을 보였다.

<div align="center">
  
![결과](https://github.com/SeungW/Roblox-Play-Video-Text-Analysis/assets/108673913/ff4e157b-6ef9-4098-9b55-f70578eef667)

</div>

> 평균 성능이 가장 높은 MaxPooling 기법으로 진행한 모든 f1
score 값에 대해 Wilcoxon rank sum test를 진행한 결과, 다른 모든 기법 대비 매우 유의한 기법으로 해석된다.(하단 오른쪽 그림)
>
> 초당 샘플링 개수를 정하는 하이퍼 파라미터 21개의 𝑁에서,
평균 성능이 가장 높은 𝑁 = 125일 때의 f1 score 결과
값에 대해 Wilcoxon rank sum test를 동일하게 진행하였다.
검정 결과, 21개의 𝑁 중 과반수 이상의 비교에서 유의한
결과가 도출되었다.(하단 왼쪽 그림)




<div align="center">

![유의성 검정](https://github.com/JihoonPark99/NLP_study/assets/108673913/7f6a52a7-1f5e-4d97-a80b-0ce03b08fe48)

</div>

























