---
title: 7.2 프로젝트 환경 구축하기
description: 네이버 Geocoding API를 활용하는 방법에 대해 설명한다.
keywords: [도로명주소,좌표계, 데이터정제, 공공데이터, geocoding, 네이버API]
url: "/chapter-7/chapter-7-2.html"
---
# 7.2 환경 구축하기

지오코딩(Geocoding)은 주소나 지명 등의 문자열 데이터를 지리 좌표(경도, 위도) 값으로 변환하는 것을 말한다. 반대로 Reverse Geocoding은 지리 좌표 값에 해당하는 주소나 지명을 얻는 과정이다.

주요 지도 서비스 업체들(네이버, 구글, 카카오 등)은 이러한 Geocoding과 Reverse Geocoding 기능을 API 형태로 제공한다. 이를 활용하면 데이터에 있는 주소 정보를 좌표로 변환하거나, 반대로 좌표 데이터에서 주소를 얻을 수 있다.

대부분의 Geocoding API는 주소 문자열을 입력받아 해당 지점의 좌표와 더불어 행정구역, 지번, 도로명 주소 등의 정보를 제공한다. 또한 Reverse Geocoding 기능도 함께 지원하여 편리하게 활용할 수 있다.

이 장에서 사용하는 데이터와 코드 원본은 아래 링크에서 확인할 수 있다.

- 데이터: [GitHub](https://github.com/hike-lab/address-data-guide/tree/main/chapter-7/data)
- 코드: [GitHub](https://github.com/hike-lab/address-data-guide/blob/main/chapter-7/7-2_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_%ED%99%98%EA%B2%BD%EC%84%A4%EC%A0%95.ipynb)

## 네이버 클라우드 API 발급/활용 방법
### 네이버 클라우드 접속

API를 활용하기 위해 API Key를 먼저 발급받아야 한다. 네이버 지오코딩 API Key는 [네이버 클라우드 플랫폼](https://www.ncloud.com/)에서 발급받는다. 

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-n-cloud.png" title="naver cloud main page">
</figure>


로그인 후, 화면 상단의 콘솔 버튼을 클릭한다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-console.png" title="click console">
</figure>


버튼을 클릭하시면 다음과 같은 화면이 나타난다. 좌측 내비게이션에서 `Services` 항목을 클릭한다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-console-main.png" title="click console">
</figure>

카테고리에서 `AI·NAVER API` 항목을 클릭한다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-services.png" title="click console">
</figure>

### APP 등록

해당 항목의 페이지로 이동하면 API 사용량에 대한 대시보드가 나타난다. 여기에서 `Application 등록` 버튼을 클릭한다.
<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-api-page.png" title="click console">
</figure>

등록 화면에서 Application 이름을 작성하고 service를 선택하면 등록 버튼이 활성화된다. Application 이름은 API 키들을 구분할 용도로 앱 이름을 설정하는 것이니 자유롭게 작성하면 된다. 이용할 Service로는 Maps의 Geocoding과 Revese Geocoding을 선택한다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-app-enroll.png" title="click console">
</figure>

등록이 완료된 후 첫 화면으로 돌아오면 대시보드에 방금 전 등록한 앱이 있는 것을 확인하실 수 있다.

이제 API도 발급받았으니, 활용에 필요한 Key 정보를 확인한다. Key 정보를 보기 위해 앱 이름 하단에 있는 `인증 정보` 버튼을 클릭한다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-key-button.png" title="click console">
</figure>

인증 정보 버튼을 누르면 앱의 key에 대한 정보를 담은 팝업창이 나타난다. 팝업창은 앱 이름과 함께 Client ID, Client Secret 값이 나오는데, 여기서 `Client Secret`은 외부에 노출되면 안되는 값이다. 만일 값이 노출되었다면 재발급 버튼을 눌러서 키를 재발급 받아야 한다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-key-popup.png" title="click console">
</figure>

key 값은 메모장에 복사해두고 사용하면 된다. 권장하는 방법은 .env 파일에 저장하여 사용하는 것이다. 특히, 깃허브를 통해 코드를 외부에 공유할 경우, API 키 값들을 모두 env 파일에 저장하고 코드 파일은 API 명칭으로만 불러와서 사용할 수 있기 때문에 key값이 노출되지 않는다는 장점이 있다. 또한 키 값을 작성하고 지우는 과정이 생략되어 매우 용이하다. 

향후 실습에서 .env 파일로 key를 불러와 사용하도록 하는 방식이므로, 어떻게 .env 파일을 만들고 사용하는지, 그리고 `.gitignore` 파일을 작성하여 `.env` 파일은 push 되지 않도록 하는 방법을 간단하게 소개한다.

### .env, .gitignore 파일을 활용해 안전하게 key 이용하기

> colab 환경을 기준으로 설명한다. 

시작에 앞서, 코드와 데이터, 로컬 환경에서 .env, .gitignore를 한 곳에 담을 폴더를 생성하고, `새로 만들기` - `서식 있는 텍스트`를 클릭한다.

생성된 새 파일의 확장자(.rtf)를 `.env`로 변경하고, .env 파일을 메모장으로 연다. 메모장을 열었을 때 `{\rtf1}`가 이미 입력되어 있는데, 모두 삭제하고 아래 이미지와 같이 수정한다. 

`CLIENT_SECRET`는 앞서 발급 받은 Secret key, `CLIENT_ID`는 Client ID를 복사해서 그대로 붙여 넣는다. 작성이 완료되면 저장하고 창을 닫는다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-envfile.png" title="click console">
</figure>

.gitignore 파일도 동일한 방식으로 생성한 후, 메모장을 이용해 연다. .gitignore 파일에는 `*.env`를 기입하고 저장한다.

이제 colab에 접속해 새 노트를 생성한 후, 좌측 내비게이션에서 폴더 아이콘을 클릭한다. 폴더 창에 방금 만든 .env 파일과 .gitignore 파일을 폴더 창에 드롭한다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-colab.png" title="click console">
</figure>


키 값을 불러오기 위해선 `dotenv`라이브러리를 설치받아야 한다.
다운로드 코드는 아래와 같이 작성할 수 있다.

```python
!pip install python-dotenv
```

.env 파일에서 저장했던 API Key와 ID를 불러오는 코드는 다음과 같다.

```python
from dotenv import load_dotenv
import os

load_dotenv(".env")

API_ID = os.getenv("CLIENT_ID")
API_SECRET = os.getenv("CLIENT_SECRET")
```

### API 사용해보기

API를 이용하기 위해서는 requests 라이브러리가 필요하다. colab 환경에서는 이미 설치되어 있으니 ``import requests``로 간단하게 라이브러리를 불러오기만 하면 된다.

API를 사용할 때에는, API를 제공하는 측에서 정해준 형식대로 요청 URL을 작성해야 한다. 네이버의 경우, [네이버 geocode API 가이드](https://api.ncloud-docs.com/docs/ai-naver-mapsgeocoding-geocode)에 요청 URL과 파라미터, 헤더와 응답값에 대해 자세히 설명하고 있다.

아래의 코드는 간단하게 API를 테스트 할 수 있다.

```python
import requests as re
import json

# 요청 헤더에는 API 키와 아이디 값을 입력
headers = {"X-NCP-APIGW-API-KEY-ID":API_ID, "X-NCP-APIGW-API-KEY":API_SECRET} 

# 파라미터에는 검색할 주소를 입력
params = {"query" : "서울특별시 동작구 흑석로 84", "output":"json"}

# 정보를 요청할 url
url ="https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode" 

data = re.get(url, headers=headers, params=params)

# 리턴 값 확인
data.text
```
실행해보면 다음과 같은 결과를 얻을 수 있다. 역지오코딩의 경우, 요청 url와 파라미터 형식이 바뀌므로 아래 코드와 같이 수정한다.

```python
# 요청 헤더에는 API 키와 아이디 값을 입력
headers = {"X-NCP-APIGW-API-KEY-ID":API_ID, "X-NCP-APIGW-API-KEY":API_SECRET} 

# 파라미터에는 변환할 좌표계를 입력합니다. "경도,위도" 순으로 입력
params = {"coords" : "126.9573779,37.5048875", "output":"json", "orders":"roadaddr,addr"}

# 정보를 요청할 url
url ="https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc"

data = re.get(url, headers=headers, params=params)

# 리턴 값 확인
data.text
```

데이터 타입이 string 인 것을 확인할 수 있다. 원하는 값을 쉽게 가져올 수 있도록 딕셔너리(json) 형태로 변환해야 한다. 여기에는 json 라이브러리가 필요하다.

```python
import json

json_ob = json.loads(data.text)
```

변환된 결과를 보면 결과값이 아래와 같은 형식으로 구성되어 있다는 것을 파악할 수 있다. 도로명주소 뿐만 아니라, 도로명주소에 대응되는 지번주소, 영문주소, 주소구성요소와 좌표계 정보도 제공한다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-api-return.png" title="click console">
</figure>

좌표계에 대한 정보는 `addresses`키를 통해 접근할 수 있다. `addresses`에 대응하는 값은 리스트인데, 리스트에 요소가 하나뿐이므로 `json_ob["addresses"][0]`로 주소정보들을 담고 있는 딕셔너리에 접근할 수 있다. 

좌표계에 대한 정보는 "x",  "y" 키 값으로 얻는다. x는 경도, y는 위도에 대응한다. 따라서 각 값은 `json_ob["addresses"][0]["x"]`와 `json_ob["addresses"][0]["y"]`로 추출한다.

다음은 리턴된 결과로부터 좌표계를 추출하는 과정을 한 번에 나타낸 코드이다.

```python
import json

json_ob = json.loads(data.text)

lon = json_ob["addresses"][0]["x"] # 경도
lat = json_ob["addresses"][0]["y"] # 위도

```

다음은 역지오코딩 후 도로명주소, 지번주소를 추출하는 과정이다. 역지오코딩의 경우, 리턴된 주소값을 조합해야 한다. 리턴 값은 아래 사진과 같은 형식으로 구성되어 있다.

<figure class="flex flex-col items-center justify-center">
    <img src="../img/4-1-return-vals.png" title="click console">
</figure>

지오코딩에서 좌표계 데이터를 추출했던 것과 같은 방식으로 도로명주소, 지번주소 구성요소 딕셔너리에 접근할 수 있다.

```python
# 도로명주소 구성요소 딕셔너리
roadaddr = json_ob["results"][0]

# 지번주소 구성요소 딕셔너리
addr = json_ob["results"][1]

```

역지오코딩을 통해 얻은 값을 조합하는 방법은 뒷장에서 자세히 다룬다. 응답값의 키들이 각각 무엇을 의미하는지 확인하려면 [API 활용 문서](https://api.ncloud-docs.com/docs/ai-naver-mapsreversegeocoding-gc)를 참고하면 된다. 


## 필요 라이브러리 다운받기

colab 환경의 경우 런타임마다 라이브러리를 다운받아야 한다. 따라서, 코드 문서에서는 모두 코드 실행에 앞서 필요 라이브러리를 다운받을 수 있는 코드를 작성한다. 하지만 매 실행마다 라이브러리를 다운받는 것은 비효율적이므로, 로컬 사용자의 경우 가급적 라이브러리를 전역적으로 설치하는 것을 권장한다. 만약 visual studio code 환경에서 실행중이라면, 터미널 상에 아래의 코드들을 입력하여 한번에 라이브러리를 다운받을 수 있다.

```python
pip install plotly, dotenv, matplotlib, json
```
