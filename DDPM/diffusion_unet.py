import torch
import torch.nn as nn




class DownSampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels,t_emb_dim,down_sample,num_heads):
        super().__init__()
        self.down_sample=down_sample
        self.resnet_conv_first=nn.Sequential(
            nn.GroupNorm(8,in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        )
        self.t_emb_layer=nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channels)
        )
        self.resnet_conv_second=nn.Sequential(
            nn.GroupNorm(8,out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        )
        self.attention_norm=nn.GroupNorm(8,out_channels)
        self.attn=nn.MultiheadAttention(out_channels,num_heads,batch_first=True)
        self.residual_input_conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.down_sample_conv=nn.Conv2d(out_channels,out_channels,kernel_size=4,stride=2,padding=1) if self.down_sample else nn.Identity()

    def forward(self,x,t_emb):
        out=x
        resnet_input =out
        out = self.resnet_conv_first(out)
        out = out+ self.t_emb_layer(t_emb)[:,:,None,None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)
        
        batch_size,channels,h,w=out.shape
        in_attn = out.reshape(batch_size,channels,h*w).transpose(1,2)
        out_attn, _ = self.attn(in_attn,in_attn,in_attn)
        out_attn = out_attn.transpose(1,2).reshape(batch_size,channels,h,w)
        out = out+out_attn

        out = self.down_sample_conv(out)

        return out
    

class MidBlock(nn.Module):
    def __init__(self,in_channels,out_channels,t_emb_dim,n_heads):
        super().__init__()
        self.first_resnet_conv= nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8,in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8,out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
            )
        ])
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim,out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim,out_channels)
            )
        ])

        self.second_resnet_conv= nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8,out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8,out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
            )
        ])

        self.attn_norm = nn.GroupNorm(8,out_channels)
        self.attn=nn.MultiheadAttention(out_channels,n_heads,batch_first=True)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.Conv2d(out_channels,out_channels,kernel_size=1)
        ]
        )


    def forward(self,x,t_emb):
        out=x
        resnet_input =out
        out = self.first_resnet_conv[0](out)
        out = out+ self.t_emb_layers[0](t_emb)[:,:,None,None]
        out = self.second_resnet_conv[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        batch_size,channels,h,w=out.shape
        in_attn = out.reshape(batch_size,channels,h*w).transpose(1,2)
        out_attn, _ = self.attn(in_attn,in_attn,in_attn)
        out_attn = out_attn.transpose(1,2).reshape(batch_size,channels,h,w)
        out = out+out_attn

        resnet_input =out
        out = self.first_resnet_conv[1](out)
        out = out+ self.t_emb_layers[1](t_emb)[:,:,None,None]
        out = self.second_resnet_conv[1](out)
        out = out + self.residual_input_conv[1](resnet_input)
               

        return out
    


class UpSampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels,t_emb_dim,up_sample,num_heads):
        super().__init__()
        self.up_sample=up_sample
        self.resnet_conv_first=nn.Sequential(
            nn.GroupNorm(8,in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        )
        self.t_emb_layer=nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channels)
        )
        self.resnet_conv_second=nn.Sequential(
            nn.GroupNorm(8,out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        )
        self.attention_norm=nn.GroupNorm(8,out_channels)
        self.attn=nn.MultiheadAttention(out_channels,num_heads,batch_first=True)
        self.residual_input_conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.up_sample_conv=nn.ConvTranspose2d(in_channels//2,in_channels//2,kernel_size=4,stride=2,padding=1) if self.up_sample else nn.Identity()

    def forward(self,x,out_dowm,t_emb):

        x=self.up_sample_conv(x)#upsamples to half the channels
        x= torch.cat([x,out_dowm],dim=1)

        #resnet
        out=x
        resnet_input =out
        out = self.resnet_conv_first(out)
        out = out+ self.t_emb_layer(t_emb)[:,:,None,None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)
        #self attn
        batch_size,channels,h,w=out.shape
        in_attn = out.reshape(batch_size,channels,h*w).transpose(1,2)
        #print("in_attn shape:",in_attn.shape)
        out_attn, _ = self.attn(in_attn,in_attn,in_attn)
        out_attn = out_attn.transpose(1,2).reshape(batch_size,channels,h,w)
        out = out+out_attn

        return out
    
class UNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.down_channels=[32,64,128,256]
        self.mid_channels=[256,256,128]
        self.t_emb_dim=128
        self.down_sample=[True,True,False]
        self.up_sample=list(reversed(self.down_sample))

        #to get initial timestep processed, call it before any forward op in unet
        self.t_proj=nn.Sequential(
            nn.Linear(self.t_emb_dim,self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim,self.t_emb_dim)
        )

        #to convert initial images to required no of channels
        self.conv_input=nn.Conv2d(in_channels,self.down_channels[0],kernel_size=3,padding=1)

        self.downs=nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownSampleBlock(self.down_channels[i],self.down_channels[i+1],self.t_emb_dim,self.down_sample[i],num_heads=4))


        self.mids=nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.mid_channels[i],self.mid_channels[i+1],self.t_emb_dim,n_heads=4))
        
        self.ups=nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpSampleBlock(self.down_channels[i]*2,self.down_channels[i-1] if i!=0 else 16,self.t_emb_dim,self.down_sample[i],num_heads=4))

        self.norm_out=nn.GroupNorm(8,16)
        self.conv_out=nn.Conv2d(16,in_channels,kernel_size=3,padding=1)

    def get_time_embedding(self, time_steps, t_emb_dim):
        # time_steps: (B,)
        half_dim = t_emb_dim // 2

        device = time_steps.device
        time_steps = time_steps.float()

        factor = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device)) *
            torch.arange(half_dim, device=device) / half_dim
        )  # (D/2,)

        # (B, 1) * (1, D/2) → (B, D/2)
        t = time_steps[:, None] * factor[None, :]

        t_emb = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)  # (B, D)

        return t_emb



    def forward(self,x,t):
        print(x.shape)
        out=self.conv_input(x)
        print("Input to UNet:", out.shape)
        t_emb=self.get_time_embedding(t,self.t_emb_dim)
        #print(t_emb.shape)

        t_emb=self.t_proj(t_emb)

        down_outs=[]
        for down in self.downs:
            print(out.shape)
            down_outs.append(out)
            out=down(out,t_emb)


        for mid in self.mids:
            print(out.shape)
            out=mid(out,t_emb)

        for up in self.ups:
            down_out=down_outs.pop()
            print(out.shape,down_out.shape)
            out=up(out,down_out,t_emb)
        
        out=self.norm_out(out)
        out=nn.SiLU()(out)
        out=self.conv_out(out)
        print("Output of UNet:",out.shape)
        return out
    
            

            









