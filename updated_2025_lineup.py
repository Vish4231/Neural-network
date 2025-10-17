#!/usr/bin/env python3
"""
Updated 2025 F1 Driver Lineup
Confirmed driver changes and team compositions for 2025 season
"""

def get_2025_f1_lineup():
    """
    Get the confirmed 2025 F1 driver lineup with all major changes
    """
    
    lineup_2025 = {
        'Red Bull Racing': {
            'drivers': ['Max Verstappen', 'Yuki Tsunoda'],
            'numbers': [1, 30],
            'changes': 'Yuki Tsunoda promoted from Racing Bulls to replace Sergio Perez'
        },
        'Ferrari': {
            'drivers': ['Charles Leclerc', 'Lewis Hamilton'],
            'numbers': [16, 44],
            'changes': 'Lewis Hamilton moves from Mercedes to replace Carlos Sainz'
        },
        'Mercedes': {
            'drivers': ['George Russell', 'Andrea Kimi Antonelli'],
            'numbers': [63, 77],
            'changes': 'Andrea Kimi Antonelli (F2 champion) replaces Lewis Hamilton'
        },
        'McLaren': {
            'drivers': ['Lando Norris', 'Oscar Piastri'],
            'numbers': [4, 81],
            'changes': 'No changes - same lineup as 2024'
        },
        'Aston Martin': {
            'drivers': ['Fernando Alonso', 'Lance Stroll'],
            'numbers': [14, 18],
            'changes': 'No changes - same lineup as 2024'
        },
        'Alpine': {
            'drivers': ['Pierre Gasly', 'Franco Colapinto'],
            'numbers': [10, 21],
            'changes': 'Franco Colapinto (F2 graduate) replaces Esteban Ocon'
        },
        'Williams': {
            'drivers': ['Alexander Albon', 'Carlos Sainz'],
            'numbers': [23, 55],
            'changes': 'Carlos Sainz moves from Ferrari to replace Logan Sargeant'
        },
        'Racing Bulls': {
            'drivers': ['Liam Lawson', 'Isack Hadjar'],
            'numbers': [40, 50],
            'changes': 'Formerly RB/AlphaTauri - Liam Lawson and Isack Hadjar (F2 graduates)'
        },
        'Haas': {
            'drivers': ['Esteban Ocon', 'Oliver Bearman'],
            'numbers': [31, 87],
            'changes': 'Esteban Ocon moves from Alpine, Oliver Bearman (F2 champion) replaces Kevin Magnussen'
        },
        'Sauber': {
            'drivers': ['Nico Hulkenberg', 'Gabriel Bortoleto'],
            'numbers': [27, 35],
            'changes': 'Gabriel Bortoleto (F3 champion) replaces Zhou Guanyu'
        }
    }
    
    return lineup_2025

def get_major_2025_changes():
    """
    Get summary of major changes for 2025 season
    """
    
    changes = {
        'Driver Moves': [
            'Lewis Hamilton: Mercedes → Ferrari',
            'Carlos Sainz: Ferrari → Williams',
            'Esteban Ocon: Alpine → Haas',
            'Yuki Tsunoda: Racing Bulls → Red Bull Racing'
        ],
        'New Drivers': [
            'Andrea Kimi Antonelli (Mercedes) - F2 champion',
            'Oliver Bearman (Haas) - F2 champion',
            'Franco Colapinto (Alpine) - F2 graduate',
            'Gabriel Bortoleto (Sauber) - F3 champion',
            'Liam Lawson (Racing Bulls) - F2 graduate',
            'Isack Hadjar (Racing Bulls) - F2 graduate'
        ],
        'Team Changes': [
            'RB/AlphaTauri → Racing Bulls (rebrand)',
            'Sauber remains Sauber (Audi takeover delayed)'
        ],
        'Driver Departures': [
            'Sergio Perez (Red Bull) - contract not renewed',
            'Logan Sargeant (Williams) - replaced by Sainz',
            'Kevin Magnussen (Haas) - replaced by Bearman',
            'Zhou Guanyu (Sauber) - replaced by Bortoleto'
        ]
    }
    
    return changes

def create_2025_lineup_dataframe():
    """
    Create a pandas DataFrame with the 2025 lineup for use in predictions
    """
    import pandas as pd
    
    lineup_data = []
    
    for team, info in get_2025_f1_lineup().items():
        for i, driver in enumerate(info['drivers']):
            lineup_data.append({
                'driver_name': driver,
                'team_name': team,
                'driver_number': info['numbers'][i],
                'year': 2025,
                'is_rookie': driver in ['Andrea Kimi Antonelli', 'Oliver Bearman', 
                                      'Franco Colapinto', 'Gabriel Bortoleto', 
                                      'Liam Lawson', 'Isack Hadjar'],
                'previous_team': get_previous_team(driver) if driver in ['Lewis Hamilton', 'Carlos Sainz', 'Esteban Ocon', 'Yuki Tsunoda'] else team
            })
    
    return pd.DataFrame(lineup_data)

def get_previous_team(driver):
    """
    Get the previous team for drivers who moved in 2025
    """
    previous_teams = {
        'Lewis Hamilton': 'Mercedes',
        'Carlos Sainz': 'Ferrari',
        'Esteban Ocon': 'Alpine',
        'Yuki Tsunoda': 'Racing Bulls'
    }
    return previous_teams.get(driver, 'Unknown')

def print_2025_lineup_summary():
    """
    Print a summary of the 2025 F1 lineup
    """
    print("🏁 2025 F1 Driver Lineup Summary")
    print("=" * 50)
    
    lineup = get_2025_f1_lineup()
    changes = get_major_2025_changes()
    
    print("\n📊 Team Lineups:")
    print("-" * 30)
    
    for team, info in lineup.items():
        print(f"\n{team}:")
        for i, driver in enumerate(info['drivers']):
            print(f"  {info['numbers'][i]:2d}. {driver}")
        if info['changes'] != 'No changes - same lineup as 2024':
            print(f"  📝 {info['changes']}")
    
    print("\n🔄 Major Changes:")
    print("-" * 20)
    
    for category, items in changes.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print(f"\n📈 Statistics:")
    print(f"  • Total drivers: {sum(len(info['drivers']) for info in lineup.values())}")
    print(f"  • New drivers: {len(changes['New Drivers'])}")
    print(f"  • Driver moves: {len(changes['Driver Moves'])}")
    print(f"  • Teams with changes: {sum(1 for info in lineup.values() if info['changes'] != 'No changes - same lineup as 2024')}")

if __name__ == "__main__":
    print_2025_lineup_summary()
    
    # Create and save lineup DataFrame
    df = create_2025_lineup_dataframe()
    df.to_csv('2025_f1_lineup.csv', index=False)
    print(f"\n💾 2025 lineup saved to: 2025_f1_lineup.csv")
