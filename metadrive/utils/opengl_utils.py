import ctypes
import os
from enum import Enum
from typing import Union

import numpy as np
import torch
from cuda import cudart
from OpenGL import GL as gl
from OpenGL.GL import shaders

from metadrive import CONSOLE
from metadrive.utils.dotdict import dotdict


def FORMAT_CUDART_ERROR(err):
    from cuda import cudart

    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def CHECK_CUDART_ERROR(args):
    from cuda import cudart

    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(FORMAT_CUDART_ERROR(err))

    return ret


def common_opengl_options():
    # Use program point size
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    # Performs face culling
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glCullFace(gl.GL_BACK)

    # Performs alpha trans testing
    # gl.glEnable(gl.GL_ALPHA_TEST)
    try:
        gl.glEnable(gl.GL_ALPHA_TEST)
    except gl.GLError:
        pass

    # Performs z-buffer testing
    gl.glEnable(gl.GL_DEPTH_TEST)
    # gl.glDepthMask(gl.GL_TRUE)
    gl.glDepthFunc(gl.GL_LEQUAL)
    # gl.glDepthRange(-1.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # Enable some masking tests
    gl.glEnable(gl.GL_SCISSOR_TEST)

    # Enable this to correctly render points
    # https://community.khronos.org/t/gl-point-sprite-gone-in-3-2/59310
    # gl.glEnable(gl.GL_POINT_SPRITE)  # MARK: ONLY SPRITE IS WORKING FOR NOW
    try:
        gl.glEnable(gl.GL_POINT_SPRITE)  # MARK: ONLY SPRITE IS WORKING FOR NOW
    except gl.GLError:
        pass
    # gl.glEnable(gl.GL_POINT_SMOOTH) # MARK: ONLY SPRITE IS WORKING FOR NOW

    # # Configure how we store the pixels in memory for our subsequent reading of the FBO to store the rendering into memory.
    # # The second argument specifies that our pixels will be in bytes.
    # gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)


def load_shader_source(file: str = "splat.frag"):
    # Ideally we can just specify the shader name instead of an variable
    if not os.path.exists(file):
        file = f"{os.path.dirname(__file__)}/shaders/{file}"
    if not os.path.exists(file):
        file = file.replace("shaders/", "")
    if not os.path.exists(file):
        raise RuntimeError(f"Shader file: {file} does not exist")
    with open(file, "r") as f:
        return f.read()


def use_gl_program(program: Union[shaders.ShaderProgram, dict]):
    if isinstance(program, dict):
        # Recompile the program if the user supplied sources
        program = dotdict(program)
        program = shaders.compileProgram(
            shaders.compileShader(program.VERT_SHADER_SRC, gl.GL_VERTEX_SHADER),
            shaders.compileShader(program.FRAG_SHADER_SRC, gl.GL_FRAGMENT_SHADER),
        )
    return gl.glUseProgram(program)


class Mesh:
    class RenderType(Enum):
        POINTS = 1
        LINES = 2
        TRIS = 3
        QUADS = 4  # TODO: Support quad loading
        STRIPS = 5

    # Helper class to render a mesh on opengl
    # This implementation should only be used for debug visualization
    # Since no differentiable mechanism will be added
    # We recommend using nvdiffrast and pytorch3d's point renderer directly if you will to optimize these structures directly

    def __init__(
        self,
        verts: torch.Tensor = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),  # need to call update after update
        faces: torch.Tensor = torch.tensor([[0, 1, 2]]),  # need to call update after update
        colors: torch.Tensor = None,
        normals: torch.Tensor = None,
        scalars: dotdict[str, torch.Tensor] = dotdict(),
        render_type: RenderType = RenderType.TRIS,
        # Misc info
        name: str = "mesh",
        filename: str = "",
        visible: bool = True,
        # Render options
        shade_flat: bool = False,  # smooth shading
        point_radius: float = 0.015,
        render_normal: bool = False,
        # Storage options
        store_device: str = "cpu",
        compute_device: str = "cuda",
        vert_sizes=[3, 3, 3],  # pos + color + norm
        # Init options
        est_normal_thresh: int = 100000,
        # Ignore unused input
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = name
        self.visible = visible
        self.render_type = render_type

        self.shade_flat = shade_flat
        self.point_radius = point_radius
        self.render_normal = render_normal

        self.store_device = store_device
        self.compute_device = compute_device
        self.vert_sizes = vert_sizes

        self.est_normal_thresh = est_normal_thresh

        # Uniform and program
        self.compile_shaders()
        self.uniforms = dotdict()  # uniform values

        # Before initialization
        self.max_verts = 0
        self.max_faces = 0

        # OpenGL data
        if filename:
            self.load_from_file(filename)
        else:
            self.load_from_data(verts, faces, colors, normals, scalars)

    def compile_shaders(self):
        try:
            self.mesh_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source("mesh.vert"), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source("mesh.frag"), gl.GL_FRAGMENT_SHADER),
            )
            self.point_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source("point.vert"), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source("point.frag"), gl.GL_FRAGMENT_SHADER),
            )
        except Exception as e:
            print(e)
            raise e

    @property
    def n_verts_bytes(self):
        return len(self.verts) * self.vert_size * self.verts.element_size()

    @property
    def n_faces_bytes(self):
        return len(self.faces) * self.face_size * self.faces.element_size()

    @property
    def verts_data(self):  # a heavy copy operation
        verts = torch.cat([self.verts, self.colors, self.normals], dim=-1).ravel().numpy()  # MARK: Maybe sync
        verts = np.asarray(verts, dtype=np.float32, order="C")
        return verts

    @property
    def faces_data(self):  # a heavy copy operation
        faces = self.faces.ravel().numpy()  # N, 3
        faces = np.asarray(faces, dtype=np.uint32, order="C")
        return faces

    @property
    def face_size(self):
        return self.render_type.value

    @property
    def vert_size(self):
        return sum(self.vert_sizes)

    def use_gl_program(self, program: shaders.ShaderProgram):
        use_gl_program(program)
        self.uniforms.H = gl.glGetUniformLocation(program, "H")
        self.uniforms.W = gl.glGetUniformLocation(program, "W")
        self.uniforms.n = gl.glGetUniformLocation(program, "n")
        self.uniforms.f = gl.glGetUniformLocation(program, "f")
        self.uniforms.P = gl.glGetUniformLocation(program, "P")
        self.uniforms.K = gl.glGetUniformLocation(program, "K")
        self.uniforms.V = gl.glGetUniformLocation(program, "V")
        self.uniforms.M = gl.glGetUniformLocation(program, "M")
        self.uniforms.VM = gl.glGetUniformLocation(program, "VM")
        self.uniforms.focal = gl.glGetUniformLocation(program, "focal")
        self.uniforms.principal = gl.glGetUniformLocation(program, "principal")
        self.uniforms.basisViewport = gl.glGetUniformLocation(program, "basisViewport")

        if hasattr(self, "shade_flat"):
            self.uniforms.shade_flat = gl.glGetUniformLocation(program, "shade_flat")
        if hasattr(self, "point_radius"):
            self.uniforms.point_radius = gl.glGetUniformLocation(program, "point_radius")
        if hasattr(self, "render_normal"):
            self.uniforms.render_normal = gl.glGetUniformLocation(program, "render_normal")

    def update_gl_buffers(self):
        # Might be overwritten
        self.resize_buffers(
            len(self.verts) if hasattr(self, "verts") else 0, len(self.faces) if hasattr(self, "faces") else 0
        )  # maybe repeated

        if hasattr(self, "verts"):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.n_verts_bytes, self.verts_data)  # hold the reference
        if hasattr(self, "faces"):
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferSubData(gl.GL_ELEMENT_ARRAY_BUFFER, 0, self.n_faces_bytes, self.faces_data)

    def resize_buffers(self, v: int = 0, f: int = 0):
        if v > self.max_verts or f > self.max_faces:
            if v > self.max_verts:
                self.max_verts = v
            if f > self.max_faces:
                self.max_faces = f
            self.init_gl_buffers(v, f)

    def init_gl_buffers(self, v: int = 0, f: int = 0):
        element_size = (
            self.verts.element_size() if hasattr(self, "verts") else (4 if self.vert_gl_types[0] == gl.GL_FLOAT else 2)
        )

        # This will only init the corresponding buffer object
        n_verts_bytes = v * self.vert_size * element_size if v > 0 else self.n_verts_bytes

        # Housekeeping
        if hasattr(self, "vao"):
            gl.glDeleteVertexArrays(1, [self.vao])
            gl.glDeleteBuffers(2, [self.vbo, self.ebo])

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, n_verts_bytes, ctypes.c_void_p(0), gl.GL_DYNAMIC_DRAW
        )  # NOTE: Using pointers here won't work

        # https://stackoverflow.com/questions/67195932/pyopengl-cannot-render-any-vao
        cumsum = 0
        for i, (s, t) in enumerate(zip(self.vert_sizes, self.vert_gl_types)):
            gl.glVertexAttribPointer(
                i, s, t, gl.GL_FALSE, self.vert_size * element_size, ctypes.c_void_p(cumsum * element_size)
            )  # we use 32 bit float
            gl.glEnableVertexAttribArray(i)
            cumsum += s

        if hasattr(self, "faces"):
            n_faces_bytes = f * self.face_size * self.faces.element_size() if f > 0 else self.n_faces_bytes
            if n_faces_bytes > 0:
                # Some implementation has no faces, we dangerously ignore ebo here, assuming they will never be used
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, n_faces_bytes, ctypes.c_void_p(0), gl.GL_DYNAMIC_DRAW)
                gl.glBindVertexArray(0)


class Quad(Mesh):
    # A shared texture for CUDA (pytorch) and OpenGL
    # Could be rendererd to screen using blitting or just drawing a quad
    def __init__(self, H: int = 256, W: int = 256, use_quad_cuda: bool = True):  # the texture to blip
        self.use_quad_cuda = use_quad_cuda
        self.vert_sizes = [3]  # only position
        self.vert_gl_types = [gl.GL_FLOAT]  # only position
        self.render_type = Mesh.RenderType.STRIPS  # remove side effects of settings _type
        self.max_verts, self.max_faces = 0, 0
        self.verts = torch.as_tensor(
            [
                [-1.0, -1.0, 0.5],
                [1.0, -1.0, 0.5],
                [-1.0, 1.0, 0.5],
                [1.0, 1.0, 0.5],
            ]
        )
        self.H, self.W = H, W
        self.update_gl_buffers()
        self.compile_shaders()
        self.init_texture()

    @property
    def n_faces_bytes(self):
        return 0

    def use_gl_program(self, program: shaders.ShaderProgram):
        super().use_gl_program(program)
        self.uniforms.tex = gl.glGetUniformLocation(program, "tex")
        gl.glUseProgram(self.quad_program)  # use a different program
        gl.glUniform1i(self.uniforms.tex, 0)

    def compile_shaders(self):
        try:
            self.quad_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source("quad.vert"), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source("quad.frag"), gl.GL_FRAGMENT_SHADER),
            )
        except Exception as e:
            print(e)
            raise e

    def resize_textures(self, width: int, height: int):  # analogy to update_gl_buffers
        self.H, self.W = height, width
        self.init_texture()

    def init_texture(self):
        if hasattr(self, "cu_tex"):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_tex))

        if hasattr(self, "tex"):
            gl.glDeleteTextures(1, [self.tex])

        # Init the texture to be blit onto the screen
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, self.W, self.H, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, ctypes.c_void_p(0)
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        if self.use_quad_cuda:
            try:
                flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
                self.cu_tex = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.tex, gl.GL_TEXTURE_2D, flags))
            except (RuntimeError, ModuleNotFoundError) as e:
                CONSOLE.log(f"[bold red]Failed to initialize Quad with CUDA-GL interop, will use slow upload: {e}")
                self.use_quad_cuda = False

    def copy_to_texture(self, image: torch.Tensor):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).cuda()
        image = torch.flip(image, [0]).contiguous()
        if not self.use_quad_cuda:
            self.upload_to_texture(image, 0, 0)
            return
        image = image[: self.H, : self.W]

        if image.shape[-1] == 3:
            image = torch.cat([image, image.new_ones(image.shape[:-1] + (1,)) * 255], dim=-1)
        h, w = image.shape[:2]
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.cu_tex, torch.cuda.current_stream().cuda_stream))
        cu_tex_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(self.cu_tex, 0, 0))

        CHECK_CUDART_ERROR(
            cudart.cudaMemcpy2DToArrayAsync(
                cu_tex_arr,
                0,
                0,
                image.data_ptr(),
                w * 4 * image.element_size(),
                w * 4 * image.element_size(),
                h,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                torch.cuda.current_stream().cuda_stream,
            )
        )
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.cu_tex, torch.cuda.current_stream().cuda_stream))

    def upload_to_texture(self, ptr: np.ndarray, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        if isinstance(ptr, torch.Tensor):
            ptr = ptr.detach().cpu().numpy()  # slow sync and copy operation # MARK: SYNC

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D, 0, x, y, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, ptr
        )  # to gpu, might slow down?

    @property
    def verts_data(self):  # a heavy copy operation
        verts = self.verts.ravel().detach().cpu().numpy()  # MARK: Maybe sync
        verts = np.asarray(verts, dtype=np.float32, order="C")
        return verts

    def render(self):
        self.draw()  # no uploading needed

    def draw(self):
        """
        Upload the texture instead of the camera
        This respects the OpenGL convension of lower left corners
        """

        _, _, W, H = gl.glGetIntegerv(gl.GL_VIEWPORT)

        gl.glUseProgram(self.quad_program)  # use a different program
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.verts))
        gl.glBindVertexArray(0)
