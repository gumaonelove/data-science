# Сборный проект 1
## Описание проекта
Заказчик этого исследования — Министерство культуры Российской Федерации. 

Вам нужно изучить рынок российского кинопроката и выявить текущие тренды. Уделите внимание фильмам, которые получили 
государственную поддержку. Попробуйте ответить на вопрос, насколько такие фильмы интересны зрителю. 

Вы будете работать с данными, опубликованными на портале открытых данных Министерства культуры. Набор данных содержит 
информацию о прокатных удостоверениях, сборах и государственной поддержке фильмов, а также информацию с сайта КиноПоиск.
## Инструкция по выполнению
### Шаг 1. Откройте файлы с данными и объедините их в один датафрейм
Объедините данные таким образом, чтобы все объекты из датасета mkrf_movies обязательно вошли в получившийся датафрейм. 
Пути к файлам: 
`/datasets/mkrf_movies.csv` — данные о прокатных удостоверениях.
`/datasets/mkrf_shows.csv` — данные о прокате в российских кинотеатрах.

#### 5.1. Общая информация о Датасетах
**Описание столбцов**
`title` — название фильма;
`puNumber` — номер прокатного удостоверения;
`show_start_date` — дата премьеры фильма;
`type` — тип фильма;
`film_studio` — студия-производитель;
`production_country` — страна-производитель;
`director` — режиссёр;
`producer` — продюсер;
`age_restriction` — возрастная категория;
`refundable_support` — объём возвратных средств государственной поддержки;
`nonrefundable_support` — объём невозвратных средств государственной поддержки;
`financing_source` — источник государственного финансирования;
`budget` — общий бюджет фильма;
`ratings` — рейтинг фильма на КиноПоиске;
`genres` — жанр фильма.
`box_office` — сборы в рублях.

**Есть пропуски**
`director`
`producer`
`refundable_support`
`nonrefundable_support`
`budget`
`financing_source`
`ratings`
`genres`
### Шаг 2. Предобработка данных
Проверьте типы данных в датафрейме и преобразуйте там, где это необходимо.

Изучите пропуски в датафрейме. Объясните, почему заполнили пропуски определённым образом или почему не стали это делать.

Проверьте, есть ли в данных дубликаты. Опишите причины, которые могли повлиять на появление дублей.

Изучите столбцы, которые содержат категориальные значения:

Посмотрите, какая общая проблема встречается почти во всех категориальных столбцах;

Исправьте проблемные значения в поле type.

Изучите столбцы, которые хранят количественные значения. Проверьте, обнаружились ли в таких столбцах подозрительные
данные. Как с такими данными лучше поступить?

Добавьте новые столбцы:
* Создайте столбец с информацией о годе проката. Выделите год из даты премьеры фильма;
* Создайте два столбца: с именем и фамилией главного режиссёра и основным жанром фильма. В столбцы войдут первые 
значения из списка режиссёров и жанров соответственно;

#### Выполнил
1. Нормальное распределение имеет столбец рэйтинга
2. Колонки `title`, `film_studio`, `director`, producer оставим типа `object`, тк для эффективности перехода к 
категориальному типу данных, необходимо менее 50% уникальных значений
3. Заполним пропуски (государственное финансирование) у фильмов (-1).
4. Удалим фильмы без рэйтинга
5. PS, как оказалось потом, кинофильмы без бюджета и информации о кассовых сборах в аналитике участвовать не будут
НО я заморочился и сегментировано заполнил пропуски
6. Явные дубликаты не обнаружены
7. Не явные дубликаты устранены

Посчитайте, какую долю от общего бюджета фильма составляет государственная поддержка.
### Шаг 3. Проведите исследовательский анализ данных
Посмотрите, сколько фильмов выходило в прокат каждый год. 
Обратите внимание, что данные о прокате в кинотеатрах известны не для всех фильмов. Посчитайте, какую долю составляют 
фильмы с указанной информацией о прокате в кинотеатрах. Проанализируйте, как эта доля менялась по годам. 
Сделайте вывод о том, какой период полнее всего представлен в данных.
Изучите, как менялась динамика проката по годам. В каком году сумма сборов была минимальной? А максимальной?
С помощью сводной таблицы посчитайте среднюю и медианную сумму сборов для каждого года. 
Сравните значения и сделайте выводы.
Определите, влияет ли возрастное ограничение аудитории («6+», «12+», «16+», «18+» и т. д.) на сборы фильма в прокате 
в период с 2015 по 2019 год? Фильмы с каким возрастным ограничением собрали больше всего денег в прокате?
Меняется ли картина в зависимости от года? Если да, предположите, с чем это может быть связано.

#### Исследовательский анализ данных
1. Доля государственной поддержки составляет ~50%
2. Мы видим резкое падения количества фильмов в 2011 и в 2017 годах
3. У половины фильмов нет информации о прокате в кинотеатрах, не учитываем их в аналитики
4. Динамика выручки по годам
   * Мы видим минимальную выручку в 2010-2012 годах
   * Резкий рост выручки в 2014-2016 годах
   * Стагнацию выручки в 2018 - 2019 годах
   * Максимальная выручка была достигнута в 2019 году
5. Динамика средней (медианной) выручки по годам - На графике мы видим резкий рост средней выручки с 2013 по 2017 год, 
но медиана практически неизменна, это значит что в года роста средней выручки были сверх успешные фильмы
6. Больше всего в сумме выручка была у фильмов с возрастным ограничением 16+
7. Фильмы с возрастным ограничением в 6 и 12 лет имеют наибольшую среднюю выручку
8. С 2016 по 2018 годы мы видим ~симметрию графика суммы выручки для фильмов с возрастными ограничениями 12+ и 16+, 
у меня есть предположение, что 14-15 дети подросли и начали смотреть фильмы 16+
9. Так же мы видим постоянные рост выручки у фильмов 18+
10. Выручка фильмов 0+ стагнирует, 6+ медленно растет
### Шаг 4. Исследуйте фильмы, которые получили государственную поддержку
На этом этапе нет конкретных инструкций и заданий — поищите интересные закономерности в данных. 
Посмотрите, сколько выделяют средств на поддержку кино. Проверьте, хорошо ли окупаются такие фильмы, какой у них рейтинг. 

#### Вывод
1. Сумма Гос поддержки фильмов увеличивается с каждым годом
2. ROS = 92%, фильмы окупаются практически в 2 раза
3. Мы видим тенденцию роста рентабельности кинофильмов, с 2013 по 2017 года, максимум был достугнут в 2017, затем наблюдается спад
4. На графике финансированя фильмов мы видим, повышение сумму инвестиций в кино с 2014 по 2016 года, а на графике динамики измениея рэйтинга по годам мы видим критическое снижение среднего (медианного) рэйтинга фильмов в 2016 году
5. Можем сделать вывод, что рэйтинг фильма не зависит от государственного финансирования, что странно.
6. На гистограмме столбца show_start_date — дата премьеры фильма, мы видим резкое увеличение числа фильмов с 2014 по 2016 года, что соответствует увеличению финансирования кино индустрии в эти года, но как мы помним роста в среднем рейтинге в эти года не было, а наоборот наблюдался спад, это значит что наши ребятя решили брать количеством, а не качеством.
7. бюджет более чем на половину зависит от государственных инвестиций
8. Рентабельность зависит от выручки (ROS введенная мной метрика)
9. Выручка зависит от объема гос инвестиций


## Описание данных
### Таблица `mkrf_movies` содержит информацию из реестра прокатных удостоверений. 
У одного фильма может быть несколько прокатных удостоверений. 
* `title` — название фильма;
* `puNumber` — номер прокатного удостоверения;
* `show_start_date` — дата премьеры фильма;
* `type` — тип фильма;
* `film_studio` — студия-производитель;
* `production_country` — страна-производитель;
* `director` — режиссёр;
* `producer` — продюсер;
* `age_restriction` — возрастная категория;
* `refundable_support` — объём возвратных средств государственной поддержки;
* `nonrefundable_support` — объём невозвратных средств государственной поддержки;
* `financing_source` — источник государственного финансирования;
* `budget` — общий бюджет фильма;
* `ratings` — рейтинг фильма на КиноПоиске;
* `genres` — жанр фильма.

Обратите внимание, что столбец `budget` уже включает в себя полный объём государственной поддержки. 

Данные в этом столбце указаны только для тех фильмов, которые получили государственную поддержку. 
### Таблица mkrf_shows содержит сведения о показах фильмов в российских кинотеатрах.
* `puNumber` — номер прокатного удостоверения;
* `box_office` — сборы в рублях.