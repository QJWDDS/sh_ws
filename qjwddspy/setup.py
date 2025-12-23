from setuptools import find_packages, setup

package_name = 'qjwddspy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shuai',
    maintainer_email='shuai@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_py = qjwddspy.test_py:main',
            'gazebo_imageball_dec_pub = qjwddspy.gazebo_imageball_dec_pub:main',\
            'gazebo_redball_dec_pub = qjwddspy.gazebo_redball_dec_pub:main',
            'vision_control_ddpg = qjwddspy.vision_control_ddpg:main',
            'vision_control_e2e = qjwddspy.vision_control_e2e:main',
        ],
    },
)
