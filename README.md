# MoodTracker :sob:

## Introduction

최근 코로나로 인하여 비대면 활동이 증가하면서 1인 가구의 사회적 고립이 심화되고 있다고 합니다. 또한 이로 인해 외로움, 고립감, 우울감과 스트레스 등의 수치가 매우 높아졌다고 합니다. 이러한 증상을 '코로나 우울증' 혹은 '코로나 블루'라고 표현하는데 이를 치료하기 위한 방법 중 하나가 일기를 작성하는 것이라고 합니다. 따라서 이번 프로젝트에서는 일기를 작성하게 되면 작성자의 감성을 분석하여 알려줄 수 있는 모델을 제작하고 이를 통해 자신도 알 지 못했던 자신의 감정에 대해 이해할 수 있도록 돕고자 합니다.

## Data

https://aihub.or.kr/aidata/7978

데이터는 'AIhub'에서 제공하는 '감성 대화 말뭉치 소개' 데이터를 사용하였습니다.

## Method

모델은 마지막 시간에 배웠던 Transformer의 인코더만을 사용하는 BERT 기반의 모델을 사용하였습니다. 현재 BERT 기반의 다양한 모델들이 존재하는데, 이번 프로젝트에서는 SK T-Brain에서 만든 KoBERT를 fine-tuning하여 사용하였습니다.

SK T-Brain https://github.com/SKTBrain/KoBERT <br>
monologg https://github.com/monologg/KoBERT-Transformers<br>
<br>
알고리즘은 아래의 링크와 영상을 보고 공부하였습니다.<br>
<br>
딥 러닝을 이용한 자연어 처리 입문<br> 02) 버트(Bidirectional Encoder Representations from Transformers, BERT)
<br>
https://wikidocs.net/115055 <br>
<br>
고려대학교 산업경영공학부 DSBA 연구실<br>
08-5: BERT<br>
https://www.youtube.com/watch?v=IwtexRHoWG0
<br>

## Process

tool은 jupyter notebook을 사용하였고, 각 과정을 하나의 .ipynb 파일로 진행하였습니다. fine-tuning 과정은 KoBERT 깃허브를 참고하였고 BERT를 직접 사용해본다는 것에 의의를 두고 진행하였습니다.

### Ver. 1

처음에는 원본 데이터에서 간단하게 필요한 상태로 변환하고 그대로 감정을 소분류 기준으로 총 58개의 class로 분류하는 모델을 만들었습니다.<br>
이 때, test 기준으로 정확도 50% 정도의 성능을 보여주었습니다.<br>

![ver_1_accuracy](https://user-images.githubusercontent.com/83542989/170863516-6e65f0e6-646a-4942-a45c-0271fb26f4e5.jpg)

### Ver. 2

성능이 조금 부족하다고 판단하였고 Confusion Matirx를 기반으로 잘못 분류한 문장들을 확인하였습니다. <br>

![경계가 애매](https://user-images.githubusercontent.com/83542989/170864544-1499a7b8-94e1-451f-bff4-69e5448f1dcc.jpg)

다음과 같이 사람이 분류를 하더라도 어려울 수 있는 부분이라고 판단하였기 때문에 이것들을 병합하여 class의 수를 47개로 줄인 후에 진행하였습니다. <br>
이 때, test 기준으로 정확도 55% 정도의 성능을 보여주었습니다.

### Ver. 3

55%에서 정확도를 더 높이기 위해 대분류로 1차 분류를 하고, 이후에 소분류를 시도하였습니다. 즉 대분류 모델 1개, 소분류 모델 6개를 학습하였습니다. 하지만 대분류 70%, 소분류에서 70~80% 정도의 성능을 보여주었기 때문에 Ver. 2보다 성능이 좋아졌다고 판단하기에는 어려움이 있었습니다. 오히려 Ver. 2와 성능이 유사하게 나왔기 때문에 감정을 labeling 하는 과정에서 완벽하게 감정이 하나에 속하기 애매하기 때문에 발생하는 결과라고 판단하였습니다.<br>

대분류 성능<br>
![대분류 성능](https://user-images.githubusercontent.com/83542989/170864751-91a58f0f-ef8a-408e-8990-9ba68d71b676.jpg)

소분류(기쁨) 성능<br>
![소분류 성능](https://user-images.githubusercontent.com/83542989/170864949-22454c26-c255-457b-b4b1-f9c279adfde5.jpg)

## Result

<img width="498" alt="image" src="https://user-images.githubusercontent.com/83542989/170871236-0b7c72d2-8612-4d62-a57c-6c7243e2e5be.png">

직접 만든 분류기로 테스트 해 본 결과입니다. 생각보다 속도도 빠르고 성능도 좋다는 생각이 들었습니다.

## Conclusion

성능이 55%이기 때문에 어떻게 보면 높은 성능이 아니라고 판단할 수 있지만, class가 47개였기 때문에 사람이 분류하더라도 애매한 경우가 많았습니다. 즉, 완전하게 다른 클래스로 잘못 분류하였다기 보다는 애초에 label이 조금 경계가 모호하다는 생각을 하였습니다. BERT 모델의 성능이 정말 우수하다는 생각과 함께 사용하기에도 편리하다고 생각하였습니다.
