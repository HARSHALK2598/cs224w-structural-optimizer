r"""A small Kaggle API helper. The real token is added by the environment's
network proxy; the token below is only a placeholder the SDK requires."""
from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
    ApiSaveKernelRequest,
)

USER = 'hsk8171'
client = KaggleClient(env=KaggleEnv.PROD,
                      api_token='KGAT_placeholder0000000000000000')
api = client.kernels.kernels_api_client


def push(slug, code, gpu=True, internet=True, kernel_type='script',
         title=None):
    req = ApiSaveKernelRequest()
    req.slug = f'{USER}/{slug}'
    req.new_title = title or slug
    req.text = code
    req.language = 'python'
    req.kernel_type = kernel_type
    req.is_private = True
    req.enable_gpu = gpu
    req.enable_internet = internet
    res = api.save_kernel(req)
    return dict(ref=res.ref, url=res.url, version=res.version_number,
                error=res.error)


def status(slug):
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = USER
    req.kernel_slug = slug
    res = api.get_kernel_session_status(req)
    return dict(status=str(res.status), failure=res.failure_message)


def output(slug):
    req = ApiListKernelSessionOutputRequest()
    req.user_name = USER
    req.kernel_slug = slug
    return api.list_kernel_session_output(req)
