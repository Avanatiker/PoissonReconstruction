use std::collections::HashMap;
use crate::geometry::{Point3, Vec3};
use crate::octree::{self, OctreeNode};
use crate::sparse::{MatrixEntry, SparseMatrix};
use crate::solvers;

pub const DIM: usize = 3;

#[derive(Debug, Clone)]
pub struct OrientedPoint { pub position: Point3, pub normal: Vec3 }

pub struct FEMTree {
    pub max_depth: u32,
    pub fem_node_count: usize,
    pub octree: octree::Octree,
    pub offset_to_idx: HashMap<[u32; DIM], usize>,
    pub solution: Vec<f64>,
    pub normal_field: Vec<Vec3>,
}

impl FEMTree {
    pub fn new(max_depth: u32) -> Self {
        FEMTree { max_depth, fem_node_count: 0, octree: octree::Octree::new(),
            offset_to_idx: HashMap::new(), solution: Vec::new(), normal_field: Vec::new() }
    }

    pub fn initialize_from_points(&mut self, points: &[OrientedPoint], threshold: f64) {
        let res = 1usize << self.max_depth; let h = 1.0 / res as f64;
        let mut cell_counts: HashMap<[u32; DIM], f64> = HashMap::new();
        for pt in points {
            let mut o = [0u32; DIM]; let mut ok = true;
            for d in 0..DIM { let c = (pt.position[d]/h) as isize; if c<0||c>=res as isize{ok=false;break;} o[d]=c as u32; }
            if ok { *cell_counts.entry(o).or_insert(0.0) += 1.0; }
        }
        let maxd = self.max_depth as usize;
        let mut dc: Vec<HashMap<[u32;DIM],f64>> = (0..=maxd).map(|_| HashMap::new()).collect();
        dc[maxd] = cell_counts;
        for d in (0..maxd).rev() {
            let mut m = HashMap::new();
            for (&[cx,cy,cz],&c) in &dc[d+1] { *m.entry([cx>>1,cy>>1,cz>>1]).or_insert(0.0) += c; }
            dc[d] = m;
        }
        fn refine(node: &mut OctreeNode, d: u32, off: [u32; DIM], mx: u32, dc: &[HashMap<[u32;DIM],f64>], th: f64) {
            if d>=mx {return;}
            let cnt = dc[d as usize].get(&off).copied().unwrap_or(0.0);
            if cnt<=th {return;}
            node.init_children();
            if let Some(ref mut kids) = node.children { for i in 0..8u32 {
                let o=[off[0]*2+(i&1),off[1]*2+((i>>1)&1),off[2]*2+((i>>2)&1)];
                refine(&mut kids[i as usize],d+1,o,mx,dc,th);
            }}
        }
        refine(&mut self.octree.root,0,[0;DIM],self.max_depth,&dc,threshold);
        if self.max_depth > self.octree.max_depth { self.octree.max_depth = self.max_depth; }
    }

    pub fn finalize(&mut self) {
        self.octree.finalize();
        let sorted = self.octree.sorted_nodes.as_ref().expect("not finalized");
        self.offset_to_idx.clear();
        let mut idx = 0;
        for &np in &sorted.tree_nodes {
            unsafe {
                let n = &mut *np;
                if n.is_leaf() && n.depth == self.max_depth {
                    n.data_mut().set_flag(octree::flags::SPACE_FLAG, true);
                    n.data_mut().set_flag(octree::flags::FEM_FLAG_1, true);
                    n.data_mut().node_index = idx as octree::NodeIndex;
                    self.offset_to_idx.insert(n.offset, idx);
                    idx += 1;
                }
            }
        }
        self.fem_node_count = idx;
        self.solution.resize(idx, 0.0);
    }

    pub fn splat_normal_field(&mut self, points: &[OrientedPoint]) {
        let n = self.fem_node_count; self.normal_field.resize(n, Vec3::ZERO);
        let res = 1usize << self.max_depth; let h = 1.0 / res as f64;
        for pt in points {
            let cx=(pt.position.x/h) as isize; let cy=(pt.position.y/h) as isize; let cz=(pt.position.z/h) as isize;
            let tx=pt.position.x/h-cx as f64; let ty=pt.position.y/h-cy as f64; let tz=pt.position.z/h-cz as f64;
            let bx0=1.0-tx; let bx1=tx; let by0=1.0-ty; let by1=ty; let bz0=1.0-tz; let bz1=tz;
            let nx=-pt.normal.x; let ny=-pt.normal.y; let nz=-pt.normal.z;
            for dz_i in 0..=1isize{for dy_i in 0..=1isize{for dx_i in 0..=1isize{
                let ox=(cx+dx_i) as u32; let oy=(cy+dy_i) as u32; let oz=(cz+dz_i) as u32;
                if ox>=res as u32||oy>=res as u32||oz>=res as u32{continue;}
                if let Some(&idx)=self.offset_to_idx.get(&[ox,oy,oz]) {
                    let bx=if dx_i==0{bx0}else{bx1}; let by=if dy_i==0{by0}else{by1}; let bz=if dz_i==0{bz0}else{bz1};
                    let w=bx*by*bz;
                    if idx<n { self.normal_field[idx].x+=nx*w; self.normal_field[idx].y+=ny*w; self.normal_field[idx].z+=nz*w; }
                }
            }}}
        }
    }

    fn m1(d: isize)->f64{match d.abs(){0=>2./3.,1=>1./6.,_=>0.}}
    fn k1(d: isize)->f64{match d.abs(){0=>2.,1=>-1.,_=>0.}}
    fn dm(d: isize)->f64{match d{0=>0.,1=>-0.5,-1=>0.5,_=>0.}}

    pub fn assemble_system_matrix(&self, points: &[OrientedPoint], _d: u32, pw: f64) -> SparseMatrix<f64> {
        let n = self.fem_node_count; let res=1usize<<self.max_depth; let h=1.0/res as f64;
        let i2o:Vec<[u32;DIM]>={let mut v=vec![[0u32;DIM];n];for(&o,&i) in &self.offset_to_idx{v[i]=o;}v};
        let mut re:Vec<Vec<(usize,f64)>>=vec![Vec::new();n];
        for i in 0..n{let[ix,iy,iz]=i2o[i];
            for dx in -1isize..=1isize{for dy in -1isize..=1isize{for dz in -1isize..=1isize{
                let jx=ix as isize+dx;let jy=iy as isize+dy;let jz=iz as isize+dz;
                if jx<0||jx>=res as isize||jy<0||jy>=res as isize||jz<0||jz>=res as isize{continue;}
                if let Some(&j)=self.offset_to_idx.get(&[jx as u32,jy as u32,jz as u32]){
                    let v=Self::k1(dx)*Self::m1(dy)*Self::m1(dz)+Self::m1(dx)*Self::k1(dy)*Self::m1(dz)+Self::m1(dx)*Self::m1(dy)*Self::k1(dz);
                    let val=v*h; if val!=0.0{re[i].push((j,val));}
                }
            }}}
        }
        if pw>0.0 { let wh3=pw*h.powi(3);
            for pt in points {
                let cx=(pt.position.x/h) as isize;let cy=(pt.position.y/h) as isize;let cz=(pt.position.z/h) as isize;
                let tx=pt.position.x/h-cx as f64;let ty=pt.position.y/h-cy as f64;let tz=pt.position.z/h-cz as f64;
                let bx0=1.-tx;let bx1=tx;let by0=1.-ty;let by1=ty;let bz0=1.-tz;let bz1=tz;
                for dz_i in 0..=1isize{for dy_i in 0..=1isize{for dx_i in 0..=1isize{
                    let ox=(cx+dx_i)as u32;let oy=(cy+dy_i)as u32;let oz=(cz+dz_i)as u32;
                    if ox>=res as u32||oy>=res as u32||oz>=res as u32{continue;}
                    if let Some(&idx)=self.offset_to_idx.get(&[ox,oy,oz]){
                        let bx=if dx_i==0{bx0}else{bx1};let by=if dy_i==0{by0}else{by1};let bz=if dz_i==0{bz0}else{bz1};
                        re[idx].push((idx,wh3*(bx*by*bz).powi(2)));
                    }
                }}}
            }
        }
        for row in &mut re { row.sort_by_key(|(j,_)|*j);
            let mut j=0;let mut u=0;
            while j<row.len(){let col=row[j].0;let mut s=0.;
                while j<row.len()&&row[j].0==col{s+=row[j].1;j+=1;}
                row[u]=(col,s);u+=1;}
            row.truncate(u);
        }
        let nnz:usize=re.iter().map(|r|r.len()).sum();
        let mut mat=SparseMatrix::with_capacity(n,nnz);
        for i in 0..n{mat.set_row_size(i,re[i].len());}
        mat.finalize_structure();
        for i in 0..n{let row=mat.row_mut(i);for(k,&(col,val))in re[i].iter().enumerate(){row[k]=MatrixEntry::new(col,val);}}
        mat
    }

    pub fn assemble_rhs_from_field(&self, points: &[OrientedPoint], _d: u32, pw: f64) -> Vec<f64> {
        let n=self.fem_node_count;let res=1usize<<self.max_depth;let h=1.0/res as f64;let mut rhs=vec![0.0;n];
        let i2o:Vec<[u32;DIM]>={let mut v=vec![[0u32;DIM];n];for(&o,&i) in &self.offset_to_idx{v[i]=o;}v};
        for i in 0..n{let[ix,iy,iz]=i2o[i];let mut s=0.;
            for dx in -1isize..=1isize{for dy in -1isize..=1isize{for dz in -1isize..=1isize{
                let jx=ix as isize+dx;let jy=iy as isize+dy;let jz=iz as isize+dz;
                if jx<0||jx>=res as isize||jy<0||jy>=res as isize||jz<0||jz>=res as isize{continue;}
                if let Some(&j)=self.offset_to_idx.get(&[jx as u32,jy as u32,jz as u32]){
                    if j<self.normal_field.len(){
                        let v=self.normal_field[j];
                        s-=v.x*Self::dm(dx)*Self::m1(dy)*Self::m1(dz)+v.y*Self::m1(dx)*Self::dm(dy)*Self::m1(dz)+v.z*Self::m1(dx)*Self::m1(dy)*Self::dm(dz);
                    }
                }
            }}}
            rhs[i]=s;
        }
        if pw>0.0{let wh3=pw*h.powi(3);let t=0.5;
            for pt in points{
                let cx=(pt.position.x/h)as isize;let cy=(pt.position.y/h)as isize;let cz=(pt.position.z/h)as isize;
                let tx=pt.position.x/h-cx as f64;let ty=pt.position.y/h-cy as f64;let tz=pt.position.z/h-cz as f64;
                let bx0=1.-tx;let bx1=tx;let by0=1.-ty;let by1=ty;let bz0=1.-tz;let bz1=tz;
                for dz_i in 0..=1isize{for dy_i in 0..=1isize{for dx_i in 0..=1isize{
                    let ox=(cx+dx_i)as u32;let oy=(cy+dy_i)as u32;let oz=(cz+dz_i)as u32;
                    if ox>=res as u32||oy>=res as u32||oz>=res as u32{continue;}
                    if let Some(&idx)=self.offset_to_idx.get(&[ox,oy,oz]){
                        let bx=if dx_i==0{bx0}else{bx1};let by=if dy_i==0{by0}else{by1};let bz=if dz_i==0{bz0}else{bz1};
                        rhs[idx]+=wh3*t*bx*by*bz;
                    }
                }}}
            }
        }
        rhs
    }

    pub fn solve_cascadic(&mut self, a: &SparseMatrix<f64>, b: &[f64], gs: usize, cg: usize, eps: f64) {
        let n=a.num_rows();self.solution.resize(n,0.);self.solution.fill(0.);
        for _ in 0..gs{solvers::gauss_seidel_sweep(a,b,&mut self.solution);}
        solvers::solve_cg(a,b,&mut self.solution,cg,eps);
    }

    pub fn evaluate(&self, p: &Point3, _d: u32) -> f64 {
        let res=1usize<<self.max_depth;let h=1.0/res as f64;
        let sx=p.x/h;let sy=p.y/h;let sz=p.z/h;
        #[inline]fn co(s:f64,r:usize)->(u32,u32){let f=s.floor()as isize;let r=r as isize;((f-1).clamp(0,r-1)as u32,f.clamp(0,r-1)as u32)}
        #[inline]fn bv(s:f64,o:u32)->f64{let d=s-o as f64;if d<0.||d>2.{0.}else if d<=1.{d}else{2.-d}}
        let(ox0,ox1)=co(sx,res);let(oy0,oy1)=co(sy,res);let(oz0,oz1)=co(sz,res);
        let bx0=bv(sx,ox0);let bx1=bv(sx,ox1);let by0=bv(sy,oy0);let by1=bv(sy,oy1);let bz0=bv(sz,oz0);let bz1=bv(sz,oz1);
        let mut v=0.;
        for&(ox,bx)in&[(ox0,bx0),(ox1,bx1)]{if bx==0.{continue;}
            for&(oy,by)in&[(oy0,by0),(oy1,by1)]{if by==0.{continue;}
                for&(oz,bz)in&[(oz0,bz0),(oz1,bz1)]{if bz==0.{continue;}
                    if let Some(&idx)=self.offset_to_idx.get(&[ox,oy,oz]){if idx<self.solution.len(){v+=self.solution[idx]*bx*by*bz;}}
                }}}
        v
    }
}
