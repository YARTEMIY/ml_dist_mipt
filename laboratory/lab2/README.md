# Overview
С помощью алмазов кот Иван успешно заполучил господство над Землей. Однако оказалось, что быть властелином мира дело хлопотное и утомительное, поэтому кот Иван начал искать, чем же можно развлечься на незнакомой планете.

Кот Иван очень любит спорт. Но так как его мышцы были тяжело атрофированы после длительного космического путешествия, играть в спортивные игры вроде футбола или керлинга он совсем не мог. Оставался только один вариант — киберспорт…

В числе прочих влиятельных людей на Земле, кот Иван успел познакомиться с Габеном, от которого узнал про такую замечательную игру, как Dota2.

Короче:
Ваша задача — предсказать исход матча по Dota2, а именно вероятность победы команды Сил Света.

# Data
В качестве датасета собраны характеристики матечей по DOTA2, включающие как характеристики самой игры, так и отдельных игроков.

Список файлов:
* DOTA2_TRAIN_features.csv - Признаки матчей для обучения
* DOTA2_TRAIN_targets.csv - Исход матча, 1 если победили силы света, 0 иначе
* DOTA2_TEST_features.csv - Признаки матчей для лидерборда
* sample_submission.csv - Пример посылки
* tome_of_knowledge.jsonl - Источник силы, превращающей зверя в человека, а человека — в бога.

# Rules
Правила на этот контест такие:

* Командная работа не допускается
* Делиться кодом и списывать нельзя
* Можете выбрать любую модель, принцип работы которой будете готовы рассказать на * экзамене.
* Обучение вашей модели на платформе Google Colab должно занимать не более 2 часов (наличие вычислительных ресурсов не должно быть для вас нечестным преимуществом ;D ). Мы попросим вас добавить %%time в ячейку, запускающую обучение.
* Решение должно быть воспроизводимо (не забывайте фиксировать seed в python, numpy и torch!)