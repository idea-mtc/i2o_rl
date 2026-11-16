from pathlib import Path
from pxr import Usd,UsdPhysics,Sdf
for task in ['open_microwave','put_books_on_bookshelf']:
 p=Path('assets/rlbench_'+task+'/usd/task.usda');stage=Usd.Stage.Open(str(p));count=0
 for prim in stage.Traverse():
  if prim.HasAPI(UsdPhysics.CollisionAPI):
   prim.AddAppliedSchema('PhysxCollisionAPI');prim.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.001);prim.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
   if '/book' in str(prim.GetPath()):UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
   count+=1
  if prim.HasAPI(UsdPhysics.RigidBodyAPI):
   prim.AddAppliedSchema('PhysxRigidBodyAPI');prim.CreateAttribute('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int).Set(16);prim.CreateAttribute('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int).Set(4)
 stage.GetRootLayer().Save();print(task,count)
