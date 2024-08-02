---
title: 5.5 데이터 삽입하기
description: 도로명주소 한글 데이터를 테이블에 삽입하는 방법을 학습합니다.
keywords: [MySQL, Docker, Python, pymysql, 주소데이터분석]
url: "/chapter-5/chapter-5-5.html"
---

# 5.5 데이터 삽입하기

이번 장은 도로명주소 한글 데이터를 테이블에 삽입하는 방법을 학습한다. `local infile`을 활용해 파일 전체를 테이블로 업로드하고, 데이터가 담긴 테이블에서 일부 데이터를 추출하는 질의문까지 작성해본다.

이 장에서 사용하는 데이터와 코드 원본은 아래 링크에서 확인할 수 있다.

- 데이터: [GitHub](https://github.com/hike-lab/address-data-guide/tree/main/chapter-5/data)
- 코드: [GitHub](https://github.com/hike-lab/address-data-guide/blob/main/chapter-5/5_%ED%8C%8C%EC%9D%B4%EC%8D%AC%EA%B3%BC_MySQL%EB%A1%9C_%EA%B5%AC%EC%B6%95%ED%95%98%EB%8A%94_%EC%A3%BC%EC%86%8C_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4.ipynb)

## local infile 허용하기

`local infile`은 로컬 파일에서 MySQL로 데이터를 직접 업로드할 수 있게 하는 기능이다. 로컬에서 데이터를 직접 읽기 때문에 대용량의 데이터를 업로드할 때 속도가 향상된다. 다만 보안상의 문제가 있을 수 있기 때문에 초기 MySQL은 `local infile` 기능을 허용해두고 있지 않다. 하지만 실습할 데이터는 적은 용량이 아니기 때문에 `local infile` 기능을 사용하는 것이 효과적이다.

`local infile` 기능은 클라이언트와 서버 단에서 모두 허용해주어야 한다. 앞서 [5.2장에서 도커로 MySQL을 실행할 때](/contents/chapter-5/chapter-5-2.html#docker-mysql-실행하기), `--loose-local-infile=1`을 설정해준 적이 있다. 이는 서버 단에서 `local infile` 기능을 허용한 것이다. 클라이언트 단에서 설정해주려면, `pymysql`로 접속할 때 해당 기능을 설정할 수 있다.다음은 `init_db_connection()` 함수에 `local infile`를 허용해주는 조건을 추가한 것이다.

```py
def init_db_connection():
    connection = pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        password="root",
        database="address",
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8",
        local_infile=True # 클라이언스 상에서 조건 추가
    )
    return connection
```

## 데이터 경로 확인하기

로컬로 데이터를 올리기 위해서 데이터 경로를 확인해야 한다. 도로명주소 한글 데이터는 `data/rnaddrkor`에 모두 저장했고, 이 폴더에 전체분(`rnaddrkor`)과 관련지번(`jibun_rnaddrkor`) 데이터가 함께 포함되어 있다. 전체분부터 데이터를 `rnaddrkor` 테이블에 넣는다고 하면, 다음의 코드로 해당 데이터를 가져올 수 있다. `glob.glob("data/rnaddrkor/rnaddrkor_*.txt")`은 `rnaddrkor_`로 시작하는 파일명만 모두 가져온다. 17개의 시도별로 나누어진 데이터 목록이 출력된다.

```py
import glob

file_list = glob.glob("data/rnaddrkor/rnaddrkor_*.txt")

for file in file_list:
    print(file)
# data/rnaddrkor/rnaddrkor_sejong.txt
# data/rnaddrkor/rnaddrkor_jeonnam.txt
# data/rnaddrkor/rnaddrkor_seoul.txt
# data/rnaddrkor/rnaddrkor_ulsan.txt
# data/rnaddrkor/rnaddrkor_gyeongnam.txt
# data/rnaddrkor/rnaddrkor_incheon.txt
# data/rnaddrkor/rnaddrkor_chungnam.txt
# data/rnaddrkor/rnaddrkor_daejeon.txt
# data/rnaddrkor/rnaddrkor_daegu.txt
# data/rnaddrkor/rnaddrkor_chungbuk.txt
# data/rnaddrkor/rnaddrkor_gangwon.txt
# data/rnaddrkor/rnaddrkor_gyunggi.txt
# data/rnaddrkor/rnaddrkor_gwangju.txt
# data/rnaddrkor/rnaddrkor_jeju.txt
# data/rnaddrkor/rnaddrkor_busan.txt
# data/rnaddrkor/rnaddrkor_gyeongbuk.txt
# data/rnaddrkor/rnaddrkor_jeonbuk.txt
```

## 데이터 업로드하기

`rnaddrkor` 테이블에 시도별로 구분된 데이터를 모두 업데이트 해보자. 다음의 코드는 17개의 시도 데이터를 한번에 테이블로 업로드한다. 개별 데이터는 `cp949`로 읽고, 이를 다시 `utf8`로 작성해 임시파일로 저장하고 MySQL 테이블에 업로드한다. 코드에 대한 자세한 설명은 다음과 같다.

- 1-2줄: 필요한 모듈을 불러온다.
- 4줄: 데이터가 모두 업로드되었는지 확인하기 위해 데이터의 총 행수를 알아야한다. `total_line_count`는 총 행수를 저장할 변수다.
- 6줄: `file_list`에 저장된 데이터를 하나씩 for문으로 돌리면서 작업을 수행한다.
- 7-11줄: 도로명주소 한글은 `cp949`로 인코딩되었기 때문에 `cp949`로 데이터를 읽는다.
- 13-17줄: 파일을 하나씩 읽으면서 총 행수를 추가한다.
- 19-22줄: `cp949`로 읽은 파일을 다시 `utf8`로 인코딩된 파일에 작성하고 임시저장한다.
- 24-29줄: 임시파일에 저장한 데이터를 테이블에 업로드한다.
- 31-32줄: 업로드된 임시파일은 삭제한다.

대용량의 데이터라 데이터를 업로드하는데 시간이 걸릴 수 있다. 실습과정에서 `rnaddrkor` 테이블에 데이터를 넣는데 약 1분 40초가 소요되었고, `jibun_rnaddrkor` 테이블에 데이터를 넣는데 25초가 소요되었다.

```py
import os
from tqdm import tqdm

total_line_count = 0  # 총 행수를 저장할 변수

for file in tqdm(file_list, desc='Processing files'):
    file_path = os.path.abspath(file)

    # cp949로 파일 읽고 행 불러오기
    with open(file_path, 'r', encoding='cp949', errors='ignore') as f:
        lines = f.readlines()

    # 행 수 계산
    line_count = len(lines)

    # 총 행 수에 더하기
    total_line_count += line_count

    # cp949로 읽은 파일을 utf8로 임시파일에 저장
    temp_file_path = "temp_file.txt"
    with open(temp_file_path, 'w', encoding='utf8') as f:
        f.writelines(lines)

    sql = f'''
        LOAD DATA LOCAL INFILE "{temp_file_path}" INTO TABLE rnaddrkor
        FIELDS TERMINATED BY "|";
    '''
    print(f"Processing file: {file_path}, Number of lines: {line_count}")
    query_update(sql) # query_update로 수정

    # 임시파일 삭제
    os.remove(temp_file_path)

print(f"Total number of lines: {total_line_count}")
# Processing files: 100%|██████████| 17/17 [01:43<00:00,  6.12s/it]
# Total number of lines: 6384988
```

로컬 데이터를 MySQL 테이블에 업로드할 때 사용한 SQL을 더 자세히 살펴보자. `LOAD DATA LOCAL INFILE "{temp_file_path}"`은 데이터가 저장된 로컬 경로를 작성한다. `INTO TABLE rnaddrkor`은 데이터를 업로드할 테이블명을 작성한다. `FIELDS TERMINATED BY "|"`는 컬럼을 구분하는 구분자로 "|"를 설정한다.

```sql
LOAD DATA LOCAL INFILE "{temp_file_path}" INTO TABLE rnaddrkor \\
    FIELDS TERMINATED BY "|";
```

지금까지 코드는 전체분 데이터를 `rnaddrkor` 테이블에 업로드하는 방법이다. 관련지분 데이터를 `jibun_rnaddrkor` 테이블에 업로드하려면 (1) `file_list`에 정의된 파일 경로를 관련지번 데이터에 맞게 수정하고, (2) `sql` 변수에 담긴 테이블을 `jibun_rnaddrkor`로 수정한다. 전체 코드는 [깃헙에 공개된 파이썬 노트북](https://github.com/hike-lab/address-data-guide/blob/main/chapter-5/5_%ED%8C%8C%EC%9D%B4%EC%8D%AC%EA%B3%BC_MySQL%EB%A1%9C_%EA%B5%AC%EC%B6%95%ED%95%98%EB%8A%94_%EC%A3%BC%EC%86%8C_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4.ipynb)에서 확인할 수 있다.

## 데이터 업로드 확인하기

데이터가 MySQL에 잘 업로드되었는지 확인해보자. 기본적인 `SELECT` 구문을 활용해 데이터를 출력한다. 다음의 코드는 `rnaddrkor` 테이블에서 5행의 데이터를 출력한다. `SELECT *`는 테이블의 모든 컬럼을 가져오고, `LIMIT 5`는 5행의 데이터만 출력한다. 만약 일부 컬럼에 대한 데이터만 가져오고 싶다면, `SELECT 도로명주소관리번호 FROM rnaddrkor LIMIT 5;`와 같이 컬럼명을 작성한다.

```py
sql = "SELECT * FROM rnaddrkor LIMIT 5;"
query_get(sql)
```

다음의 코드는 `rnaddrkor` 테이블의 모든 행수를 계산하는 코드다. `COUNT(*)`은 행의 개수를 계산한다. 로컬 데이터의 총 행수가 6,384,988건이며, 테이블의 모든 행수와 동일하다. 데이터가 무사히 MySQL에 올라간 것을 확인할 수 있다.

```py
sql = "SELECT COUNT(*) FROM rnaddrkor;"
query_get(sql)
# [{'COUNT(*)': 6384988}]
```
