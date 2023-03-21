# Data Visualization

## Veri Seti
Bu veri kümesi ilk olarak UCI Makine Öğrenimi Deposu tarafından kullanıma sunulmuştur (bağlantılar: https://archive.ics.uci.edu/ml/datasets/wine+quality ).

- Google Drive ile bağlantı kurulur
```python
from google.colab import drive
drive.mount('/content/gdrive')

```
- Kütüphaneler içe aktarılır
```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

```
- "Dosya Seç" e tıklanarak bilgisayardaki data yüklenir (Veri setini yüklemek için farklı seçenekler de var)
```python
from google.colab import files
uploaded = files.upload()

```
- Veriler içe aktarılır
```python
df = pd.read_csv("winequality-red.csv",sep=";")
df.head()

```

## EDA (Exploratory Data Analysis)
Exploratory Data Analysis, özet istatistikler ve grafik gösterimler yardımıyla kalıpları keşfetmek, anormallikleri tespit etmek, hipotezi test etmek ve varsayımları kontrol etmek için veriler üzerinde ilk araştırmaları gerçekleştirmenin kritik sürecini ifade eder. Veri setini anlamlandırmamızı sağlar.

```python
#ilk 5 satırı gösterir
df.head(5)

```
```python
#Son 5 satırı satırı gösterir
df.tail()

```
```python
#Rastgele 5 satırı gösterir
df.sample(5)

```
```python
#Veri seti hakkında bilgi verir
df.info()

```
```python
# Hangi sütunda kaç tane boş veri var onu gösterir
df.isnull().sum()

```
- Bir şarabın kalitesi şeker oranı, sitrik asit oranı gibi diğer değişkenlere bağlı olarak değişecepi için bu veri setinde bağımlı değişkenimiz "quality"dir.
```python
# İstatistik verilerini gösterir 
df.describe()

```
```python
#Bir sütunun diğer bir sütuna olan bağlılığını gösterir
df.corr()

```
```python
#Korelasyonu ısı haritası kullanarak görselleştirdik
#"annot=True" kullanarak hücrelerdeki değerleri yazdırmış olduk
plt.figure(figsize=(15,6))
sns.heatmap(df.corr(),annot=True)

```
- Korelasyon "0"a yaklaştıkça değişkenlerin arasında doğrusal bir ilişki olamadığınız anlarız. Bu tabloya bakarak "quality"nin "residual sugar" ve "free sulfur dioxide" ile nerdeyse bir ilişkisinin olmadığını anlayabiliriz.

![image](https://user-images.githubusercontent.com/46057146/226600442-80e5fcd1-5767-43c5-8a38-aa0c76e262e1.png)

- Veri setinde şarabın kalitesine verilen puanları görmüş olduk
```python
# ‘unique()’ verilen verinin kaç adet ‘eşşiz’ verisi olduğunu bize verir
df["quality"].unique()

```
- Puanların verisetindeki dağılımı
```python
df.quality.value_counts()

```
- Boxplot = Verideki dağılımını, çarpık ve basıklık yönünden verileri özetlemek, ve herhangi bir aykırı değer olup olmadığını anlamak için kullanılır.
```python
sns.boxplot(x=df["residual sugar"]);

```
![image](https://user-images.githubusercontent.com/46057146/226600784-2200caf7-5af8-409f-b610-f7fb4fe41c3c.png)

```python
df.plot(kind='box',figsize=(15,6));

```
![image](https://user-images.githubusercontent.com/46057146/226600869-2cc816ba-5c8c-42a0-aefd-ad1bd2a88ce7.png)

- Grafiği anlamdırmak zor olduğu için farklı bir yöntem deniyoruz
```python
#Aykırı değerin kırmızı bir alt çizgi dairesi olarak tanımlamak için
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
"""
fig, eksenleri plt.subplots olarak tanımlanır. 
Sütun sayısı df.columns uzunluğuna eşit olacaktır. 
"""
fig, axs = plt.subplots(1, len(df.columns), figsize=(35,10))

for i, ax in enumerate(axs.flat):

    #Burada i indeks değişkenine göre sütunları alıyoruz ve aykırı özelliklerimizi belirtiriz ve red_circle'ı ekliyoruz.
    ax.boxplot(df.iloc[:,i], flierprops=red_circle)
    df.iloc[:,i]
    ax.set_title(df.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    
    #Sütun adlarının sütunlara eşit olup olmadığını kontrol ediliyor
    if df.columns[i] == 'RDEP' or df.columns[i] == 'RMED':
        ax.semilogy()
    
plt.tight_layout()

```
![image](https://user-images.githubusercontent.com/46057146/226600998-f5c63009-cd73-429c-b069-d9ba42564ab6.png)

```python
sns.pairplot(df,hue="quality")

```
![image](https://user-images.githubusercontent.com/46057146/226601038-95830bf9-bfd8-4b59-93a6-1ce7c0208fd5.png)


```python
#Alcohol ve quality arasındaki ilişkiyi incelemek için 
plt.bar(df['quality'], df['alcohol'])
plt.title('Alcohol ve quality arasındaki ilişki')
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.legend()
plt.show()

```
![image](https://user-images.githubusercontent.com/46057146/226601084-f3555643-3a7d-497e-ae82-2daa8b8c5cad.png)

```python
plt.bar(df['quality'], df['volatile acidity'])
plt.title('Volatile acidity ve quality arasındaki ilişki')
plt.xlabel('quality')
plt.ylabel('volatile acidity')
plt.legend()
plt.show()

```
![image](https://user-images.githubusercontent.com/46057146/226601139-5fd8d5d0-cf66-4076-a8a7-2c7f08396faf.png)

## Feature Engineering
Feature Engineering, ham verileri seçme, değiştirme ve denetimli öğrenmede kullanılabilecek özelliklere dönüştürme işlemidir. Eğitim setinde olmayan yeni değişkenler oluşturmak için verilerden yararlanan bir makine öğrenimi tekniğidir. Veri dönüşümlerini basitleştirmek ve hızlandırmak ve aynı zamanda model doğruluğunu artırmak amacıyla hem denetimli hem de denetimsiz öğrenme için yeni özellikler üretebilir.
- Önce veri setini normalleştiriyorum. Verilerin normalleştirilmesi, dağılımının tek tip bir aralığa sahip olması için verileri dönüştürecektir. Veri kümesinin aralıklarını 0 ile 1 arasında tek tip bir aralığa normalleştiriyorum.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
normal_df = scaler.fit_transform(df)
normal_df = pd.DataFrame(normal_df, columns = df.columns)
print(normal_df.head())

```
- Ardından, sınıflandırma sonuçlarını biraz daha doğrudan hale getirmek için, orijinal df veri kümesinde "iyi şarap" adlı yeni bir sütun oluşturdum. "İyi şarap", kalite 6'ye eşit veya üzerinde olduğunda "evet"e eşittir. "İyi şarap", kalite 6'den düşük olduğunda "hayır"a eşittir.
```python
df["excellent quality"] = ["yes" if i >= 6 else "no" for i in df['quality']]

```
**One-hot kodlama** , sonlu bir kümenin bir öğesinin o kümedeki dizinle temsil edildiği, yalnızca bir öğenin dizininin "1" olarak ayarlandığı ve diğer tüm öğelere aralık içindeki dizinlerin atandığı bir kodlama türüdür [ 0, n-1]. Her bitin 2 değeri (yani 0 ve 1) temsil edebildiği ikili kodlama şemalarının aksine, bu şema olası her durum için benzersiz bir değer atar
```python
#one hot-encoding
df["excellent quality"] = pd.get_dummies(df["excellent quality"],drop_first=True)
df["excellent quality"][:5]

```


**Kaynak**
[1] https://www.datacamp.com/blog/what-is-data-visualization-a-guide-for-data-scientists
[2]
