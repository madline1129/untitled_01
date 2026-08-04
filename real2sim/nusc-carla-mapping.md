# nuScenes → CARLA Blueprint Mapping

| nuScenes Category | CARLA Blueprint | Status |
|---|---|---|
| `noise` | — | skip |
| `animal` | `walker.pedestrian.0001` | ✓ |
| `human.pedestrian.adult` | `walker.pedestrian.0001` | ✓ |
| `human.pedestrian.child` | `walker.pedestrian.0001` | ✓ |
| `human.pedestrian.construction_worker` | `walker.pedestrian.0001` | ✓ |
| `human.pedestrian.personal_mobility` | `walker.pedestrian.0001` | ✓ |
| `human.pedestrian.police_officer` | `walker.pedestrian.0001` | ✓ |
| `human.pedestrian.stroller` | `walker.pedestrian.0001` | ✓ |
| `human.pedestrian.wheelchair` | `walker.pedestrian.0001` | ✓ |
| `movable_object.barrier` | `static.prop.streetbarrier` | ✓ |
| `movable_object.debris` | `static.prop.dirtdebris01` | ✓ |
| `movable_object.pushable_pullable` | `static.prop.bin` | ✓ |
| `movable_object.trafficcone` | `static.prop.trafficcone01` | ✓ |
| `static_object.bicycle_rack` | — | skip |
| `vehicle.bicycle` | `vehicle.bh.crossbike` | ✓ |
| `vehicle.bus.bendy` | `vehicle.carlamotors.european_hgv` | ✓ |
| `vehicle.bus.rigid` | `vehicle.carlamotors.european_hgv` | ✓ |
| `vehicle.car` | `vehicle.audi.etron` | ✓ |
| `vehicle.construction` | `vehicle.carlamotors.firetruck` | ✓ |
| `vehicle.emergency.ambulance` | `vehicle.ford.ambulance` | ✓ |
| `vehicle.emergency.police` | `vehicle.dodge.charger_police` | ✓ |
| `vehicle.motorcycle` | `vehicle.yamaha.yzf` | ✓ |
| `vehicle.trailer` | `vehicle.carlamotors.carlacola` | ✓ |
| `vehicle.truck` | `vehicle.carlamotors.european_hgv` | ✓ |
| `vehicle.ego` | `vehicle.lincoln.mkz_2017` | ✓ |
| `flat.driveable_surface` | — | terrain, not spawned |
| `flat.other` | — | terrain, not spawned |
| `flat.sidewalk` | — | terrain, not spawned |
| `flat.terrain` | — | terrain, not spawned |
| `static.manmade` | — | background, not spawned |
| `static.other` | — | background, not spawned |
| `static.vegetation` | — | background, not spawned |

## Notes

- **Terrain/background** (`flat.*`, `static.*`) are not spawned — they're implicitly represented by the CARLA map itself.
- **`static_object.bicycle_rack`** has no corresponding CARLA static prop.
- **Pedestrians** all map to `walker.pedestrian.0001`; CARLA has 51 pedestrian models (`.0001`–`.0051`) for visual variety.
- **Vehicles** pick a single representative blueprint per category; CARLA offers 41 vehicle models for diversity.
