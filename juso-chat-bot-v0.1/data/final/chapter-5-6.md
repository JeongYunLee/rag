---
title: 5.6 SQL로 주소 데이터 분석하기
description: SQL을 사용해 시도별 도로명개수를 분석하고, 도로명개수와 관련된 지번의 개수를 분석합니다.
keywords: [MySQL, Docker, Python, pymysql, 주소데이터분석]
url: "/chapter-5/chapter-5-6.html"
---

# 5.6 SQL로 주소 데이터 분석하기

이번 장은 SQL을 사용해 주소 데이터를 분석한다. 분석 과제는 총 2가지다. 첫번째 분석은 시도별 도로명주소의 개수를 분석하고, 두번째 분석은 개별 도로명주소와 관련된 지번의 개수를 분석한다.

이 장에서 사용하는 데이터와 코드 원본은 아래 링크에서 확인할 수 있다.

- 데이터: [GitHub](https://github.com/hike-lab/address-data-guide/tree/main/chapter-5/data)
- 코드: [GitHub](https://github.com/hike-lab/address-data-guide/blob/main/chapter-5/5_%ED%8C%8C%EC%9D%B4%EC%8D%AC%EA%B3%BC_MySQL%EB%A1%9C_%EA%B5%AC%EC%B6%95%ED%95%98%EB%8A%94_%EC%A3%BC%EC%86%8C_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4.ipynb)

## 분석1. 시도별 도로명주소의 개수 분석하기

시도별로 도로명주소의 개수를 분석하기 위해서 우선적으로 도로명주소를 만들어야 한다. 현재 `rnaddrkor` 테이블은 개별 도로명주소와 관련된 정보를 갖고 있고, 해당 정보를 조합해야 도로명주소를 생성할 수 있다. 생성된 도로명주소는 `full_rna_addr` 테이블에 저장하고, 해당 테이블을 활용해 시도별 도로명주소의 개수를 분석한다. 분석한 결과는 파이썬 노트북으로 가져올 수 있고, `plotly`를 사용해 시각화한다.

### 도로명주소 만들기

[5.4장의 테이블 스키마란?](/contents/chapter-5/chapter-5-4.html#테이블-스키마란)에서 설명한 활용가이드는 해당 데이터를 활용해 도로명주소를 조합하는 SQL 코드를 제공한다. 다만, 이 SQL 코드는 Oracle 기준이기 때문에 MySQL에 맞게 SQL 코드를 변경하는 것이 필요하다. 다음의 코드는 도로명주소를 조합하는 규칙을 MySQL 기반의 SQL로 수정한 것이다. 이 코드는 `rnaddrkor` 테이블에서 데이터를 가져와 `full_rna_addr` 테이블을 새롭게 생성한다.

- 2줄: `full_rna_addr`이란 테이블을 생성한다.
- 3줄: `새우편번호` 컬럼에 들어갈 값을 생성한다. `rnaddrkor` 테이블의 `기초구역번호(우편번호)` 컬럼을 가져와 `새우편번호` 컬럼의 값으로 삽입한다.
- 4-44줄: `도로명주소1` 컬럼에 들어갈 값을 생성하는 코드다. `CONCAT`으로 조건별 값을 조합한다. `도로명주소1` 컬럼은 참고항목을 포함한 도로명주소를 생성한다.
- 45-58줄: `도로명주소2` 컬럼에 들어갈 값을 생성하는 코드다. `도로명주소2`는 `도로명주소1`과 같이 `CONCAT`으로 조건별 값을 조합하고, 참고항목을 포함하지 않은 도로명주소를 생성한다.
- 59줄: `rnaddrkor` 테이블에서 `도로명주소관리번호`의 값을 가져온다.
- 60줄: `rnaddrkor` 테이블에서 `시도명`의 값을 가져온다.

```py
sql = '''
CREATE TABLE full_rna_addr AS
SELECT DISTINCT A.`기초구역번호(우편번호)` AS 새우편번호,
    CONCAT(A.시도명, ' ',
        CASE WHEN A.시군구명 = '' THEN '' ELSE CONCAT(A.시군구명, ' ') END,
        CASE WHEN A.읍면동명 = '' THEN '' ELSE
            CASE WHEN A.리명 = '' THEN '' ELSE CONCAT(A.읍면동명, ' ') END
        END,
        A.도로명, ' ',
        CASE WHEN A.지하여부 = 0 THEN ''
            WHEN A.지하여부 = 1 THEN '지하 '
            WHEN A.지하여부 = 2 THEN '공중 '
            ELSE ''
        END,
        A.건물본번,
        CASE WHEN A.건물부번 = 0 THEN '' ELSE CONCAT('-', A.건물부번) END,
        CASE
            WHEN A.공동주택구분 = '0' THEN
                CASE
                    WHEN A.읍면동명 = '' THEN ''
                    ELSE
                        CASE WHEN A.리명 = '' THEN CONCAT(' (', A.읍면동명, ')') ELSE '' END
                END
            WHEN A.공동주택구분 = '1' THEN
                CASE
                    WHEN A.읍면동명 = '' THEN
                        CASE
                            WHEN A.시군구용건물명 = '' THEN ''
                            ELSE CONCAT(' (', A.시군구용건물명, ')')
                        END
                    ELSE CONCAT(' (',
                        CASE
                            WHEN A.리명 = '' THEN CONCAT(A.읍면동명, ', ')
                            ELSE ''
                        END,
                        CASE
                            WHEN A.시군구용건물명 = '' THEN ''
                            ELSE A.시군구용건물명
                        END,
                    ')')
                END
            ELSE ''
        END
    ) AS 도로명주소1,
    CONCAT(A.시도명, ' ',
        CASE WHEN A.시군구명 = '' THEN '' ELSE CONCAT(A.시군구명, ' ') END,
        CASE WHEN A.읍면동명 = '' THEN '' ELSE
            CASE WHEN A.리명 = '' THEN '' ELSE CONCAT(A.읍면동명, ' ') END
        END,
        A.도로명, ' ',
        CASE WHEN A.지하여부 = 0 THEN ''
            WHEN A.지하여부 = 1 THEN '지하 '
            WHEN A.지하여부 = 2 THEN '공중 '
            ELSE ''
        END,
        A.건물본번,
        CASE WHEN A.건물부번 = 0 THEN '' ELSE CONCAT('-', A.건물부번) END
    ) AS 도로명주소2,
    A.도로명주소관리번호 AS 도로명주소관리번호,
    A.시도명 AS 시도명
FROM rnaddrkor A;
'''

query_update(sql)
```

`full_rna_addr` 테이블이 생성되었다면 다음의 코드로 생성된 도로명주소를 확인할 수 있다. 10개 행의 도로명주소를 출력한 결과는 다음과 같다. `도로명주소1`은 법정동과 같은 참고항목이 포함된 주소를, `도로명주소2`는 참고항목이 포함되지 않은 주소를 출력한다.

```py
sql = "SELECT * FROM full_rna_addr LIMIT 10;"
query_get(sql)
# [{'새우편번호': '03047',
#   '도로명주소1': '서울특별시 종로구 자하문로 94 (청운동)',
#   '도로명주소2': '서울특별시 종로구 자하문로 94',
#   '도로명주소관리번호': '11110101310001200009400000',
#   '시도명': '서울특별시'},
```

### 시도별 도로명주소의 개수 분석하기

이제 `full_rna_addr` 테이블로 시도별 도로명주소의 개수를 분석할 수 있다. 다음의 코드는 시도별로 도로명주소의 개수를 분석한다. `도로명주소관리번호`는 개별 도로명주소를 식별하는 식별자다. `시도명`을 기준으로 `GROUP BY`를 하고, 이를 기준으로 고유한 `도로명주소관리번호`의 개수를 `count`라는 변수로 부여한다. 이 `count`를 기준으로 내림차순 정렬을 하는 것이 코드의 결과다.

```py
sql = '''
SELECT `시도명`, COUNT(DISTINCT `도로명주소관리번호`) AS count \\
    FROM full_rna_addr \\
    GROUP BY `시도명` \\
    ORDER BY count DESC;
'''
query_get(sql)
# [{'시도명': '경기도', 'count': 1020563},
#  {'시도명': '경상북도', 'count': 713862},
#  {'시도명': '경상남도', 'count': 653480},
#  {'시도명': '전라남도', 'count': 598298},
#  {'시도명': '서울특별시', 'count': 530721},
#  {'시도명': '충청남도', 'count': 491608},
#  {'시도명': '전북특별자치도', 'count': 437063},
#  {'시도명': '강원특별자치도', 'count': 364686},
#  {'시도명': '충청북도', 'count': 335225},
#  {'시도명': '부산광역시', 'count': 301822},
#  {'시도명': '대구광역시', 'count': 229727},
#  {'시도명': '인천광역시', 'count': 186039},
#  {'시도명': '제주특별자치도', 'count': 155426},
#  {'시도명': '광주광역시', 'count': 120474},
#  {'시도명': '대전광역시', 'count': 114026},
#  {'시도명': '울산광역시', 'count': 104458},
#  {'시도명': '세종특별자치시', 'count': 27510}]
```

결과에 따르면, 경기도가 한국에서 가장 많은 도로명주소를 갖고 있다. 이에 반해, 세종특별자치시가 가장 작은 도로명주소를 갖고 있다. 이 결과를 더 효과적으로 보여주기 위해 데이터를 판다스의 데이터프레임 안에 담아 시각화를 해보자. 다음의 코드는 위의 코드와 동일한 결과를 가져오지만, `df`에 데이터프레임으로 저장된다.

```py
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:root@localhost:3307/address?charset=utf8", echo=True)

conn = engine.connect()
df = pd.read_sql(sql, con=conn)
conn.close()

df
```

이 `df`에 담긴 데이터를 파이썬 시각화 모듈인 `plotly`를 사용해 바그래프로 표현해보자. 바그래프의 x축에 `df['시도명']`으로, y축에 `df['count']`으로 표현한다.

```py
import plotly.graph_objects as go
import pandas as pd

# 바그래프 생성
bar = go.Bar(x=df['시도명'], y=df['count'], name='도로명주소관리번호 개수')

# 레이아웃 설정
layout = go.Layout(title='시도별 도로명주소의 개수')

# Figure 생성과 출력
fig = go.Figure(data=bar, layout=layout)
fig.show()
```

<embed src="/docs/guide/address-count-per-sido.html" width="100%" height="450px"></embed>

시각화한 결과, 시도별 도로명주소의 개수를 효과적으로 파악할 수 있다. 이렇게 MySQL에 저장된 데이터를 파이썬으로 조작할 수 있는 데이터프레임 형태로 만들어 파이썬이 제공하는 다양한 도구를 활용할 수 있다.

## 분석2. 관련지번이 많은 도로명주소 분석하기

다음은 도로명주소와 관련된 지번의 개수를 분석하는 과제다. 이를 분석하기 위해 전체분과 관련지번 테이블을 연계하고, 하나의 도로명주소에 붙은 관련지번의 개수를 세야 한다. 가장 많은 관련지번의 개수가 붙은 상위 10개의 도로명주소를 출력해보자.

### 관련지번 테이블에서 PNU 생성하기

관련지번 테이블은 각 행을 식별하는 PK가 도로명주소관리번호, 법정동코드, 산여부, 지번본번(번지), 지번부번(호)이다. 그러나 도로명주소관리번호를 제외해도 관련지번 테이블에서 개별 행을 구분할 수 있다. 법정동코드와 산여부, 지번본번(번지), 지번부번(호)의 조합을 PNU(Parcel NUmber)라고 한다. PNU는 법정동코드는 10자리, 산여부는 1자리, 지번본번(번지)는 4자리, 지번부번(호)는 4자리로 총 19자리이다. 이처럼 PNU는 4가지 컬럼의 값을 조합해야 한다.

PNU 컬럼을 생성하기에 앞서 여러 개의 SQL 질의를 수행할 수 있는 함수를 소개한다. `multi_query_update()` 함수는 기존의 `query_update()` 함수와 기능은 같지만, 여러 개의 SQL 질의문을 처리하기 위해 `sql_list`를 생성하고 하나씩 실행한다. 기존의 `query_update()`는 질의문을 한개씩 수행할 수 있었지만, `multi_query_update()`는 여러 개의 질의문을 연달아 수행할 수 있다.

```py
def multi_query_update(sql):
    connection = init_db_connection()
    with connection:
        with connection.cursor() as cursor:
            sql_list = [query.strip() for query in sql.strip().split('\n')]
            for query in sql_list:
                cursor.execute(query)
                print(f"Query executed: {query}")
            connection.commit()
            return True
```

이제 `jibun_rnaddrkor` 테이블에 PNU 컬럼을 생성해보자. `sql`은 2줄의 질의문을 포함한다. 첫번째 질의문은 `ALTER TABLE`을 사용해 테이블을 변경하는데, `VARCHAR(19)` 형식의 PNU 컬럼을 추가해라는 의미다. `jibun_rnaddrkor` 테이블은 PNU에 해당하는 컬럼 스키마가 존재하지 않으므로 먼저 생성해야 한다. 두번째 질의문은 `UPDATE`를 사용해 테이블을 업데이트한다. `jibun_rnaddrkor` 테이블의 PNU 값에 해당 컬럼들의 값을 모두 더하라(`CONCAT`)는 명령어다. 이때 `LPAD(`지번본번(번지)`, 4, '0')`은 `지번본번(번지)`의 값을 0을 채워서 4자리로 만들어라는 의미다. 즉, `지번본번(번지)`의 값이 2일 경우, 0002로 만드는 코드다. 따라서 PNU의 값은 모두 19자리로 만들어진다.

```py
sql = '''
ALTER TABLE jibun_rnaddrkor ADD PNU VARCHAR(19);
UPDATE jibun_rnaddrkor SET PNU = CONCAT(`법정동코드`, `산여부`, LPAD(`지번본번(번지)`, 4, '0'), LPAD(`지번부번(호)`, 4, '0'));
'''
multi_query_update(sql)
```

위의 코드 결과로 PNU가 잘 생성되었는지 확인해보자. PNU 컬럼만 가져와 결과를 출력해보면, 19자리의 PNU가 생성된다.

```py
sql = "SELECT PNU FROM jibun_rnaddrkor LIMIT 5;"
query_get(sql)
# [{'PNU': '1111010100001300003'},
#  {'PNU': '1111010100001290002'},
#  {'PNU': '1111010100001290003'},
#  {'PNU': '1111010100001310001'},
#  {'PNU': '1111010100001290001'}]
```

### 도로명주소 테이블과 관련지번 테이블 연결하기

개별 도로명주소에 붙은 관련지번의 개수를 계산하기 위해 도로명주소와 관련지번을 연계하는 것이 필요하다. `full_rna_addr` 테이블은 `도로명주소1` 또는 `도로명주소2`와 같이 도로명주소의 값을 가지고 있고, `jibun_rnaddrkor` 테이블은 관련지번의 정보를 갖고 있다. 도로명주소관리번호는 2개의 테이블을 연결하는 기준이 된다. `full_rna_addr` 테이블을 중심으로 `jibun_rnaddrkor` 테이블을 연결하는 것은 다음의 코드를 사용한다.

- 2줄: 개별 테이블에서 가져올 컬럼을 입력한다. 여기서 A는 `full_ran_addr`의 약어이고, B는 `jibun_rnaddrkor`의 약어다.
- 3줄: `full_rna_addr` 테이블이 기준이다.
- 4줄: `LEFT JOIN`을 사용해 `jibun_rnaddrkor` 테이블을 연결한다.
- 5줄: `ON`으로 연결의 기준점이 되는 `도로명주소관리번호`를 명시한다.

```py
sql = '''
SELECT A.`도로명주소관리번호`, A.`도로명주소1`, A.`도로명주소2`, B.PNU
FROM full_rna_addr A
LEFT JOIN jibun_rnaddrkor B
ON A.`도로명주소관리번호` = B.`도로명주소관리번호`
LIMIT 10;
'''
query_get(sql)
# [{'도로명주소관리번호': '11110101310001200009400000',
#   '도로명주소1': '서울특별시 종로구 자하문로 94 (청운동)',
#   '도로명주소2': '서울특별시 종로구 자하문로 94',
#   'PNU': None},
# ...
```

주석처리된 결과와 같이 위의 코드를 통해 2개의 테이블이 연계한 결과를 가져올 수 있다. 다만 PNU가 `None`으로 표현된 경우는 해당 도로명주소에 관련지번이 없다는 의미다.

그렇다면 이제 도로명주소를 기준으로 관련지번의 개수를 세보자. 우선 `도로명주소관리번호`를 기준으로 PNU의 개수를 세어본다. `도로명주소1` 또는 `도로명주소2`를 기준으로 PNU 개수를 세면 쿼리의 실행시간이 매우 오래 걸린다. 먼저 `도로명주소관리번호`를 기준으로 `GROUP BY`를 한 다음 PNU의 개수를 세고, 출력된 상위 10개의 `도로명주소관리번호`에 대해서 `도로명주소1`과 `도로명주소2`를 출력해보자. 위의 SQL 코드를 변행해서 `도로명주소관리번호`에 붙은 PNU의 개수가 많은 순대로 출력한 결과다.

```py
from sqlalchemy import create_engine
import pandas as pd

sql = '''
SELECT A.`도로명주소관리번호`, COUNT(DISTINCT B.PNU) AS PNU_COUNT
FROM full_rna_addr A
LEFT JOIN jibun_rnaddrkor B ON A.`도로명주소관리번호` = B.`도로명주소관리번호`
GROUP BY A.`도로명주소관리번호`
ORDER BY PNU_COUNT DESC
LIMIT 10;
'''

engine = create_engine("mysql+pymysql://root:root@localhost:3307/address?charset=utf8", echo=True)

conn = engine.connect()
df1 = pd.read_sql(sql, con=conn)
conn.close()
```

`df1`에 담긴 도로명주소관리번호를 `rna_addr_list`에 담아 리스트 형태로 만들어보자. `도로명주소관리번호`의 값이 해당 리스트 안에 있는 행만 출력해서 가져온 결과가 바로 `df2`다.

```py
rna_addr_list = df1['도로명주소관리번호'].tolist()

sql = f'''
SELECT * FROM full_rna_addr WHERE `도로명주소관리번호` IN {str(rna_addr_list).replace('[','(').replace(']',')')}
'''

engine = create_engine("mysql+pymysql://root:root@localhost:3307/address?charset=utf8", echo=True)

conn = engine.connect()
df2 = pd.read_sql(sql, con=conn)
conn.close()
```

이제 `df1`과 `df2`를 병합해서 관련지번의 개수와 도로명주소를 함께 보자. 판다스의 `merge`를 사용해 두개의 데이터프레임을 병합한다.

```py
merged = df1.merge(df2, on='도로명주소관리번호').sort_values('PNU_COUNT', ascending=False)
```

::: details 병합한 결과 보기

|     도로명주소관리번호     | PNU_COUNT | 새우편번호 |                          도로명주소1                           |                 도로명주소2                  |     시도명     |
| :------------------------: | :-------: | :--------: | :------------------------------------------------------------: | :------------------------------------------: | :------------: |
| 27170109300701600003300000 |    637    |   41716    |    대구광역시 서구 고성로 33 (원대동3가, 서대구센트럴자이)     |          대구광역시 서구 고성로 33           |   대구광역시   |
| 27170103422922100003000000 |    481    |   41767    | 대구광역시 서구 서대구로29길 30 (평리동, 서대구역화성파크드림) |       대구광역시 서구 서대구로29길 30        |   대구광역시   |
| 11140153310002100024600000 |    448    |   04563    |             서울특별시 중구 청계천로 246 (방산동)              |         서울특별시 중구 청계천로 246         |   서울특별시   |
| 52710330301600100150000000 |    417    |   55365    |          전북특별자치도 완주군 이서면 콩쥐팥쥐로 1500          | 전북특별자치도 완주군 이서면 콩쥐팥쥐로 1500 | 전북특별자치도 |
| 27170103314301200044600000 |    417    |   41763    |     대구광역시 서구 당산로 446 (평리동, 서대구영무예다음)      |          대구광역시 서구 당산로 446          |   대구광역시   |
| 30200133316706000022000001 |    410    |   34059    |            대전광역시 유성구 자운로 220-1 (추목동)             |       대구광역시 서구 서대구로29길 30        |   대구광역시   |
| 27170103422922100003000000 |    481    |   41767    |            전북특별자치도 완주군 이서면 혁신로 181             |   전북특별자치도 완주군 이서면 혁신로 181    | 전북특별자치도 |
| 27170103314300400023000000 |    408    |   41774    |  대구광역시 서구 문화로 230 (평리동, 서대구역반도유보라센텀)   |          대구광역시 서구 문화로 230          |   대구광역시   |
| 48840390334304500154500000 |    384    |   52455    |               경상남도 남해군 창선면 흥선로 1545               |      경상남도 남해군 창선면 흥선로 1545      |    경상남도    |
| 48250119333505100050200000 |    379    |   50811    |              경상남도 김해시 인제로 502 (삼방동)               |          경상남도 김해시 인제로 502          |    경상남도    |

:::

결과를 보면, 대구광역시에 있는 서대구센트럴자이와 서대구역화성파크드림에 해당하는 주소가 관련지번을 많이 갖고 있다. 해당 아파트부지가 상당히 넓은 부지를 갖고 있다고 해석할 수 있다. 세번째로 관련지번이 많은 주소는 서울특별시에 있는 방산시장의 주소다. 방산시장의 개별 가게마다 지번이 부여되어 해당 주소에 관련된 지번이 많다고 할 수 있다.
