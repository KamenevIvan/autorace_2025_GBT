# AutoRace Robot - GBTBoys
Робот проходит первые 2 задания
# Запуск
```bash
colcon build --packages-select autorace_core_GBTBoys
source install/setup.bash
ros2 launch autorace_core_GBTBoys autorace_core.launch.py
```

# Время
Робот проезжат выбранный участок за:
```
[INFO] [1766469948.141535259] [mission_control]: Team_name=GBTBoys 
                                Sim_time=223.850 
                                Final_time=218.060
```

# ВНИМАНИЕ
Данный пакет работает стабильно на движке ogre, на ogre 2 другой баланс цветов, работоспособность не гарантирована!
```bash
--render-engine ogre
```