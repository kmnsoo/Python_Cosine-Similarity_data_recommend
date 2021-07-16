#코싸인 유사도를 통해 데이터 모델 유사도 체크 후 가장 비슷한 데이터 추천 프로그램

#pandas, scikitlearn 라이브러리 불러오기
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('movies_metadata/movies_metadata.csv', low_memory=False)

#데이터를 20000개로 줄인다.  현재csv파일에 데이터 4만개 이상 들어있음
data = data.head(20000)

#tf-idf를 할 때 데이터에 Null 값이 들어있으면 에러가 발생한다
#널 값이 있는지 확인 ->135개 존재
print("널 값 있는지 확인->", data['overview'].isnull().sum(),"\n")


#Null 값을 처리하는 도구인 fillna()를 사용
data['overview'] = data['overview'].fillna('')
print("fillna함수 사용 후 널 값 재확인->", data['overview'].isnull().sum(),"\n")

#이제 tf-idf를 수행
tfidf = TfidfVectorizer(stop_words='english')
# overview에 대해서 tf-idf 수행
tfidf_matrix = tfidf.fit_transform(data['overview'])
print("영화 개수와 영화 표현을 위해 사용된 단어 수->", tfidf_matrix.shape, "\n")
#출력 결과 20,000개의 영화를 표현하기위해 총 47,487개의 단어가 사용되었음

#코사인 유사도 이용해서 단어간 유사도 검출
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#영화의 타이틀과 인덱스를 가진 테이블을 만듭니다. 이 중 5개만 출력
indices = pd.Series(data.index, index=data['title']).drop_duplicates()
#print(indices.head())

#이 테이블의 용도는 영화의 타이틀을 입력하면 인덱스를 리턴하기 위함이다,.
idx = indices['Father of the Bride Part II']
#print("ID->>", idx)

def get_recommendations(title, cosine_sim=cosine_sim):
    # 선택한 영화의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 영화를 가지고 연산할 수 있습니다.
    idx = indices[title]

    # 모든 영화에 대해서 해당 영화와의 유사도를 구합니다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아옵니다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 받아옵니다.
    movie_indices = [i[0] for i in sim_scores]

    # 가장 유사한 10개의 영화의 제목을 리턴합니다.
    return data['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises')
print("선택한 영화 ->The Dark Knight Rises\n")
print("선택한 영화와 유사한 10개의 영화 목록\n", get_recommendations('The Dark Knight Rises'))