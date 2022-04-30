# Insurance_tracking_MLFlow_project
Study project on tracking and logging using MLFlow

You are working on automating the processes of an insurance company. There is a task of classifying drivers, by predicting the probability of an accident of a driver in the next year, for which the driver plans to buy insurance. To do this, you use the statistics you have collected on clients.

Your ML Engineer has been kidnapped by aliens and while recruiters are looking for a new one, you need to finish the task he left behind. Add experiment markup to track the model learning process in MLFlow.

Dataset structure:

| Field | Description|
| ----- | ------ |
| driver_id | unique identifier of the driver |
| age | age of driver at time of analysis |
| sex | gender of driver |
| car_class | class of driver's car |
| driving_experience | driving experience |
| speeding_penalties | number of fines for speeding during the year |
| parking_penalties | number of parking tickets during the year |
| total_car_accident | number of accidents for the whole driving experience |
| has_car_accident | identifier of accidents in current year (target attribute [0/1]) |
