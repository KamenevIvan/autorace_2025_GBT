# AutoRace Robot - GBTBoys
Робот прохходит первые 2 задания
# Запуск
```bash
colcon build --packages-select autorace_core_GBTBoys
source install/setup.bash
ros2 launch autorace_core_GBTBoys autorace_core.launch.py
```

# ВНИМАНИЕ
Данный пакет работает стабильно на движке ogre, на ogre 2 другой баланс цветов, работоспособность не гарантирована!
```bash
--render-engine ogre
```