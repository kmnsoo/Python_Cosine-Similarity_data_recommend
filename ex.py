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
print("유사도->", cosine_sim)