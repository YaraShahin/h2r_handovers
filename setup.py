from setuptools import find_packages, setup

package_name = 'h2r_handovers'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/handover.launch.xml']),
        ('share/' + package_name + '/config', [
            'config/config.rviz',
            'config/handover_params.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yara Shahin',
    maintainer_email='yarashahinstem@gmail.com',
    description='Perception, grasp estimation and handover orchestration nodes for the H2R handover pipeline.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'hand_stabilization_node = h2r_handovers.hand_stabilization_node:main',
            'grasp_selection_node = h2r_handovers.grasp_selection_node:main',
            'handover_orchestrator = h2r_handovers.handover_orchestrator:main',
        ],
    },
)
